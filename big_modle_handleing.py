import chad_LSTM as LSTM_framework
import time_ffn as FFN_framework
import running_parity_dataset as dataset
import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(precision=1, suppress=True)

def forwards_pass(lstm,ffn,inputs):
    lstm_outputs = lstm.forwards_training(inputs)
    outputs = ffn.forwards(lstm_outputs)
    return outputs,lstm_outputs

def backwards_pass(lstm,ffn,lables,ffn_inputs,cost_func,lr,optimiser):
    cost,dlt_dh = ffn.backwards(lables,ffn_inputs,cost_func,lr)
    dl_di = lstm.pipeline_backwards(dlt_dh,lr,optimiser)
    return cost


def train(lstm,ffn,inputs,lables,iterations,batch_size,cost_func,lr,optimiser):
    cost_data = []

    for i in range(iterations):
        #make a batch
        random_indercies = np.random.choice(np.shape(inputs)[1],batch_size)
        
        input_batch = inputs[:,random_indercies,:]
        lables_batch = lables[:,random_indercies,:]

        #do the forwards and backwards pass
        ouptuts,lstm_outputs = forwards_pass(lstm,ffn,input_batch)
        cost = backwards_pass(lstm,ffn,lables_batch,lstm_outputs,cost_func,lr,optimiser)
        cost_data.append(cost)

        print(f'sample output:{ouptuts[-1,0,:]} , {lables_batch[-1,0,:]}   {int((i/iterations) * 100)}% seq_len = {np.shape(inputs)[0]}')
    return cost_data

def train_until(lstm,ffn,inputs,lables,cost_threshold,batch_size,cost_func,lr,optimiser):
    cost_data = []
    cost = cost_threshold

    while cost >= cost_threshold:
        #make a batch
        random_indercies = np.random.choice(np.shape(inputs)[1],batch_size)
        
        input_batch = inputs[:,random_indercies,:]
        lables_batch = lables[:,random_indercies,:]

        #do the forwards and backwards pass
        ouptuts,lstm_outputs = forwards_pass(lstm,ffn,input_batch)
        cost = backwards_pass(lstm,ffn,lables_batch,lstm_outputs,cost_func,lr,optimiser)
        
        cost_data.append(cost)

        print(f'sample output:{ouptuts[-1,0,:]} , {lables_batch[-1,0,:]} curr cost is: {cost} seq_len = {np.shape(inputs)[0]}')
    return cost_data


#test on parity
input_size = 1
hiddern_size = 8
output_size = 2
sizes = [hiddern_size,output_size]
act_funcs = ['softmax']

ffn = FFN_framework.ffn(sizes,act_funcs)
lstm = LSTM_framework.LSTM(input_size,hiddern_size,'tanh','tanh')

inputs,lables = dataset.generate_dataset(100,300000)

cost_threshold = 0.2 #i got this number by looking at where the cost graph stablises, i just eyballed it, it doen't really have a scientific baisis
max_seq_len = 1000
cost_data = []


#train on incriceing sequance lengths, this seems to work significantly better, 
#I think this is becasue if the network loses track grad signals from far beond that point become meaningless to an extent.
#so training on shorter lengths is much more efficient to help teach it the rule.
for i in range(max_seq_len-1):
    inputs,lables = dataset.generate_dataset(i+2,3000)#start on seq_len of 2
    cost_data.extend(train_until(lstm,ffn,inputs,lables,cost_threshold,1,'cross_entropy',0.1,'none'))

examples = len(cost_data)
cost_data = np.array(cost_data)
    

plt.plot(np.arange(0,examples),cost_data)

test_examples = 10000
test_inputs,test_lables = dataset.generate_dataset(max_seq_len,test_examples)
correct = 0
test_ouputs,_ = forwards_pass(lstm,ffn,test_inputs)
for i in range(test_examples):
    if np.argmax(test_ouputs[-1,i]) == np.argmax(test_lables[-1,i]):
        correct +=1
percentage = correct/test_examples

print(f'it got {percentage*100}% correct')

plt.show()