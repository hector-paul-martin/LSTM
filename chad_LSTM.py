import numpy as np


#handles (seq_len,batch_size,fetures) throughout

def xavier_initilisation(input_size,output_size):
        bound = np.sqrt(2.0 / (input_size + output_size))
        weights =  np.random.uniform(low=-bound, high=bound, size=(input_size, output_size))
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



class LSTM:
    def __init__(self,I_size,h_size,c_act_func,o_act_func):
        #initilise size data
        self.I_size = I_size
        self.h_size = h_size
        self.o_act_func = o_act_func

        #initilise gates
        self.forget_g = gate(I_size,h_size,'sigmoid')
        self.forget_g.baises += 1 #prevent vanishing graidients in initial training
        self.input_g = gate(I_size,h_size,'sigmoid')
        self.candidate = gate(I_size,h_size,c_act_func)
        self.output_g = gate(I_size,h_size,'sigmoid')

    #handle forwards pass
    def forwards_training(self,inputs):
        #inputs are seq_len,batch_size,input_size
        input_shape = np.shape(inputs)
        seq_len,batch_size = input_shape[0],input_shape[1]

        #initilise arrays to store values for backprop
        #gate activations, pre activations and last input(last_input has both the input and the hiddern state)
        
        #could have done this in a loop but that is more thing that could go wrong. who doesnt love a mass of initilisation anyway
        self.forget_g.z = np.zeros(shape= (seq_len, batch_size, self.h_size) )
        self.forget_g.a = np.zeros_like(self.forget_g.z)
        self.input_g.z = np.zeros_like(self.forget_g.z)
        self.input_g.a = np.zeros_like(self.forget_g.z)
        self.candidate.z = np.zeros_like(self.forget_g.z)
        self.candidate.a = np.zeros_like(self.forget_g.z)
        self.output_g.z = np.zeros_like(self.forget_g.z)
        self.output_g.a = np.zeros_like(self.forget_g.z)

        self.forget_g.last_input = np.zeros(shape=(seq_len, batch_size, (self.h_size+self.I_size)))
        self.input_g.last_input = np.zeros_like(self.forget_g.last_input)
        self.candidate.last_input = np.zeros_like(self.forget_g.last_input)
        self.output_g.last_input = np.zeros_like(self.forget_g.last_input)

        #cell state and hiddern state uses seq_len+1 to add a buffer value at timestep 0 for consitant indexing
        self.cell_state = np.zeros(shape=(seq_len+1,batch_size,self.h_size))
        self.hiddern_state = np.zeros_like(self.cell_state)

        for t in range(seq_len):
            #find all gate outputs for timesteap first
            self.forget_g.forwards(inputs[t],self.hiddern_state[t],t)
            self.input_g.forwards(inputs[t],self.hiddern_state[t],t)
            self.candidate.forwards(inputs[t],self.hiddern_state[t],t)
            self.output_g.forwards(inputs[t],self.hiddern_state[t],t)

            #then find the cell state and hiddern states
            self.cell_state[t+1] = self.cell_state[t] * self.forget_g.a[t] + self.candidate.a[t] * self.input_g.a[t]
            self.hiddern_state[t+1] = act_func(self.cell_state[t+1],self.o_act_func,False) * self.output_g.a[t]
        return self.hiddern_state[1:] # return the hiddern state(cell outputs) without the buffer

    def pipeline_backwards(self,dlt_dh,lr,optimiser):#take dlt_dh as input because i can controll output layer with an FFN.

        #get dimentions
        shape = np.shape(dlt_dh)
        seq_len,batch_size = shape[0],shape[1]

        #initilise arrays to store dl_da for all gates
        self.forget_g.dl_da = np.zeros((seq_len,batch_size,self.h_size))
        self.input_g.dl_da = np.zeros_like(self.forget_g.dl_da)
        self.candidate.dl_da = np.zeros_like(self.forget_g.dl_da)
        self.output_g.dl_da = np.zeros_like(self.forget_g.dl_da)

        #initilise dl_dh and dl_dcell
        dl_dh = np.zeros_like(dlt_dh)
        dl_dh = np.concat( (np.zeros((1, batch_size, self.h_size)) , dl_dh), axis = 0)#add a buffer timesteap for t = 0 for consistant indexing
        
        dl_dh[-1] = dlt_dh[-1]#initilise base case, at timetep -1 there are no future graidients
        
        dl_dcell = np.zeros_like(dl_dh)#buffer timesteap
        dl_dcell[-1] = dlt_dh[-1] * act_func(self.cell_state[-1],self.o_act_func,True) * self.output_g.a[-1]#initilise base case


        #find dl_dh and dl_cell for all timesteaps.
        for t in reversed(range(seq_len)):#window we iterate through is shifted one to the left to include buffer, this is becase we dont need to calculate values for dl_da for gates at timesteap T+1
            #find dl_da for all gates
            self.output_g.dl_da[t] = dl_dh[t+1] * act_func(self.cell_state[t+1],self.o_act_func,False)
            self.input_g.dl_da[t] = dl_dcell[t+1] * self.candidate.a[t]
            self.forget_g.dl_da[t] = dl_dcell[t+1] * self.cell_state[t] 
            self.candidate.dl_da[t] = dl_dcell[t+1] * self.input_g.a[t]

            if t != 0: #avoid unsessasary calulations for dl_dh[0](witch is just a buffer value)
                #find dl_dh through all the different gate paths
                ouptut_path = self.output_g.backwards(t)
                input_path = self.input_g.backwards(t)
                forget_path = self.forget_g.backwards(t)
                candidate_path = self.candidate.backwards(t)

            
                #dl_dh =  direct graidient  +  graidient from future through gates    dlt_dh doesnt have a buffer timestep, so we access dlt_dh[t-1] instead
                dl_dh[t] = dlt_dh[t-1] + ouptut_path + input_path + forget_path + candidate_path

                #dl_dcell = graidient directly from hiddern state                                                  +  graidient from furute cell state
                dl_dcell[t] = dl_dh[t] * act_func(self.cell_state[t],self.o_act_func,True) * self.output_g.a[t-1] + (dl_dcell[t+1] * self.forget_g.a[t])

        #update parameters and get dl_dI useing the dl_da values found
        dl_dI_output = self.output_g.update_params(lr,optimiser)
        dl_dI_forget = self.forget_g.update_params(lr,optimiser)
        dl_dI_input = self.input_g.update_params(lr,optimiser)
        dl_dI_candidate = self.candidate.update_params(lr,optimiser)
        
        dl_dI = dl_dI_output + dl_dI_input + dl_dI_forget + dl_dI_candidate
        
        return dl_dI
        
        

class gate:
    def __init__(self,I_size,h_size,act_func):
        self.I_size = I_size
        self.h_size = h_size
        self.act_func = act_func

        self.wheigts = np.random.normal(size=(I_size+h_size,h_size))
        
        #make hiddern wheigts orhtagonal
        self.wheigts[I_size:],_ = np.linalg.qr(self.wheigts[I_size:])

        #scale input wheights to be xavier
        self.wheigts[:I_size] *= np.sqrt(2.0 / (I_size + h_size))

        self.baises = np.zeros(shape=(1,h_size))



    def forwards(self,I,h,t):
        self.last_input[t] = np.concat((I,h),axis=1)#store this for backprop

        self.z[t] = (self.last_input[t] @ self.wheigts)  + self.baises
        self.a[t] = act_func(self.z[t],self.act_func,False)


    def backwards(self,t):#takes dl_da rentruns dl_dh[t] from the gate
        return (self.dl_da[t] * act_func(self.z[t],self.act_func,True)) @ self.wheigts[self.I_size:].T
    
    def update_params(self,lr,optimiser):#
        batch_size = np.shape(self.last_input)[1]

        dl_dI = 'placeholder'#will be used if chaining with networks befor the LSTM
        
        dl_dz = self.dl_da * act_func(self.z,self.act_func,True)#get dl_dz

        transposed_last_input = self.last_input.transpose(0,2,1)# reshape to (timesteaps,inputs,batch)
        dl_dW = np.sum(transposed_last_input @ dl_dz, axis = 0 )/batch_size #this takes arrays of size (timesteaps,inputs,batch) and (timesteaps,batch,outputs)  numpy brodcasting does batched matrix multiplication to give (timesteaps,inputs,ouputs). sum over 0th axis and divide by batch_size to get grads averaged over batch size and summed over timesteps
        
        dl_db =np.average( np.sum(dl_dz,axis=(0),keepdims=True),axis = 0)#average over batch, and sum over timesteps.
        dl_db = dl_db.reshape(1,np.shape(dl_db)[-1])#reshape to be (1,hiddern)
        #ill put an optimiser here soon :)
        self.wheigts -= dl_dW * lr
        self.baises -= dl_db * lr



        #store graidients for testing
        self.dl_db = dl_db
        self.dl_dw = dl_dW
        return dl_dI


