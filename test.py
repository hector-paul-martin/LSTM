import chad_LSTM
import time_ffn as feed_forwards
import big_modle_handleing as big
import running_parity_dataset as dataset
import numpy as np
import matplotlib.pyplot as plt

def cross_entropy(lables,output):
    cost = np.sum(-lables*np.log(output))
    return cost

def softmax_ce(lables,h):
    #do softmax
    x_shifted = h - np.max(h, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    output =  exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    #do cross entropy
    cost = np.sum(-lables*np.log(output))

    #get dc_dh
    dc_dh = output - lables

    return dc_dh,cost

def compare_numerical_grads(seq_len): #you do have to manualy set the gate that you want, Im to lazy
    #initilise a network
    I_size = 1
    hiddern_size = 5
    sizes = [5,2]
    act_funcs = ['softmax']
    lstm = chad_LSTM.LSTM(I_size,hiddern_size,'tanh','tanh')
    ffn = feed_forwards.ffn(sizes,act_funcs)

    #run forwards and backwards pass with and random inputs
    inputs,lables = dataset.generate_dataset(seq_len,1)
    
    output,lstm_output = big.forwards_pass(lstm,ffn,inputs)
    cost_analitical = big.backwards_pass(lstm,ffn,lables,lstm_output,'cross_entropy',0,'none')
    
    #store dl_dw_analitical for the weight that we wish to test
    dl_dw_analitical = lstm.input_g.dl_dw[0,0]

    #useing same inputs +- a small number on the wheigts we test run forwards pass twise and store loss
    small = 10**-6#a small number

    lstm.input_g.wheigts[0,0] += small
    output,lstm_output = big.forwards_pass(lstm,ffn,inputs)
    cost_1 = cross_entropy(lables,output)

    lstm.input_g.wheigts[0,0] -= small*2
    output,lstm_output = big.forwards_pass(lstm,ffn,inputs)
    cost_2 = cross_entropy(lables,output)
    

    dl_dw_numerical = (cost_1-cost_2) / (small*2)

    #ensure numeric stability
    if dl_dw_numerical < 10**-50:
        error = 0 #placeholder
        discarded = True
    #compare dl_dw_analit and dl_dw numeric
    else:
        error = (dl_dw_analitical - dl_dw_numerical) / dl_dw_numerical
        print(f'error is {error} with seq of len {seq_len}: analitic and numeric grads are:{dl_dw_analitical,dl_dw_numerical} ')
        discarded = False
    
    return error,discarded

def MSE(output,lables):
    cost = np.mean((output - lables)**2)#cost fucntion goes here
    dc_di = 2*(output-lables) / output.shape[-1]#cost function derivitive goes here
    return dc_di,cost


averages = []
max_seq_len = 101
iterations = 100
count = 0
count2 = 0

for j in range(1,max_seq_len):
    sumation = 0
    for i in range(iterations):
        error,discarded = compare_numerical_grads(j)
        if not discarded:
            sumation += abs(error)
            count += 1
        
    if count != 0:
        average_error = sumation/count
        averages.append(average_error)
        count2 += 1

    

averages = np.array(averages)
super_average = np.mean(averages)
print(f'average error is {super_average}')

# plot erros on a graph with timesteps
plt.plot(np.arange(1,count2+1),averages)
plt.show()
print(averages)

