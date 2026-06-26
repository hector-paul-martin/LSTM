import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(precision=1, suppress=True)

def xavier_initilisation(input_size,output_size):
        bound = np.sqrt(6 / (input_size + output_size))
        weights =  np.random.uniform(low=-bound, high=bound, size=(input_size, output_size))
        return weights

def he_initilisation(i_size,o_size,a=0.01):
    bound = np.sqrt(6/(i_size*(1+(a**2))))
    weights = np.random.uniform(low = -bound,high= bound,size=(i_size,o_size))
    return weights

def act_func(x,fucntion,derivitive,alpha=0.01):
    if fucntion == 'leaky_relu':
        if derivitive:
            return np.where(x>=0,1.0,alpha) 
        else:#normal function
            return np.where(x>=0,x,alpha*x)
        
    elif fucntion == 'sigmoid':#sigmoid
        if derivitive:
            s = act_func(x,'sigmoid',False)
            return s*(1-s)
        else:
            return 1/(1+np.exp(-x))
        
    elif fucntion == 'softmax':
        if derivitive:
            input('softmax derivitive not included')

        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    elif fucntion == 'none':
        return x
    
    elif fucntion == 'tanh':
        if derivitive:
            return 1 - np.tanh(x)**2
        else:
            return np.tanh(x)
        
    input('your function was not found')

def find_dl_dz_and_cost(lables,outputs,act_func,cost_func):
    if act_func == 'softmax' and cost_func == 'cross_entropy':#assumes one hot lables and outputs

        dl_dz = outputs - lables
        cost = np.sum(-lables*np.log(outputs))
    else:
        cost = dl_dz = 'cost function/act func combination not reconised'
        input('cost function/act func combination not reconised')
    return cost,dl_dz
       

class ffn:
    def __init__(self,sizes,act_funcs):
        if len(act_funcs)+1 != len(sizes):
            input('network sizes inconsistant')

        
        self.layers = []
        for i in range(len(act_funcs)):
            self.layers.append(layer(sizes[i],sizes[i+1],act_funcs[i]))
    
    def forwards(self,inputs):
        #input layer
        output = self.layers[0].forwards(inputs)
        #iterate through all layers
        for i in range(1,len(self.layers)):
            output = self.layers[i].forwards(output)

        return output
    
    def backwards(self,lables,inputs,cost_func,lr):#assumes forwards pass with corresponding inputs has just ran

        cost, self.layers[-1].dl_dz = find_dl_dz_and_cost(lables,self.layers[-1].a,self.layers[-1].act_func,cost_func)
        for i in reversed(range(len(self.layers)-1)):
            self.layers[i].dl_dz = (self.layers[i+1].dl_dz @ self.layers[i+1].wheights.T) * act_func(self.layers[i].z,self.layers[i].act_func,True)
        

        self.layers[0].apply_grads(inputs,lr)

        for i in range(1,len(self.layers)):
            self.layers[i].apply_grads(self.layers[i-1].a,lr)
                 
        dl_di = (self.layers[0].dl_dz @ self.layers[0].wheights.T)
        return cost,dl_di
                        

class layer:
    def __init__(self,input_size,output_size,act_func):  
        self.act_func = act_func
        self.i_size = input_size
        self.o_size = output_size
        self.act_func = act_func
        
        self.baises = np.zeros(shape=(1,output_size))

        if act_func in ['sigmoid','softmax']:
            self.wheights = xavier_initilisation(input_size,output_size)
        elif act_func == 'leaky_relu':
            self.wheights = he_initilisation(input_size,output_size)
        else:
            input(f'your act func:{act_func} is not reconised')
        
    def forwards(self,input):
        self.z = (input @ self.wheights) + self.baises
        self.a = act_func(self.z,self.act_func,False)
        return self.a
    
    def apply_grads(self,last_layer_activation,lr):#assume the data is 3 dimentional, (seq_len,batch_size,layer_size)
        
        dl_db = np.average(np.average(self.dl_dz,axis=0),axis= 0, keepdims=True)#average to (1,layer size)
        dl_dw = np.average( (np.transpose(last_layer_activation,(0,2,1)) @ self.dl_dz), axis= 0)
        
        
        self.wheights -= dl_dw * lr
        self.baises -= dl_db * lr


def test(model,inputs,lables,lr,iterations,batch_size,cuboid_size,cost_func,test_inputs,test_lables):
    cost_data = []
    for i in range(iterations):
        #make a batch
        random_indicies = np.random.choice(np.shape(inputs)[0],size = batch_size, replace=False)
        input_batch = inputs[random_indicies]
        lables_batch = lables[random_indicies]
        lables_3d = np.reshape(lables_batch,cuboid_size)
        inputs_3d = np.reshape(input_batch,cuboid_size)#reshape it into a 3d shape to simulate data with timesteap component
        output = model.forwards(inputs_3d)
        cost,dl_di =model.backwards(lables_3d,inputs_3d,cost_func,lr)
        cost_data.append(cost)

        print(f'output:{output[1,1,:]}, lable {lables_3d[1,1,:]}')

    output = model.forwards(test_inputs)
    print('testing on unseen data')
    for i in range(10):
        print(f'the prediction was{output[i]} and the lable was{test_lables[i]}')
    return np.array(cost_data)
