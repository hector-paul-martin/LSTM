import numpy as np
#我怕来不及等你回来所以 say good by


def generate_dataset(seq_len,no_of_examples):# generate a dataset that has random intergers and a running withc == [1,0] if there are an odd number of ones and [0,1] if there are an even number of ones
    #initiliseation
    inputs = np.zeros(shape=(seq_len,no_of_examples,1))
    lables = np.zeros(shape=(seq_len,no_of_examples,2))
    running_parity = np.zeros((no_of_examples,1))
    #generate inputs and expected parites
    for i in range(seq_len):
        this_timesteap = np.random.randint(low=0,high=2,size = (no_of_examples,1))
        running_parity = running_parity - this_timesteap
        running_parity = np.abs(running_parity)
        inputs[i] += this_timesteap
        lables[i] += running_parity

    #turn lables into one hot encodeing
    for i in range(len(lables)):
        lables[i] = np.where(lables[i] == 1,np.array([1,0]),np.array([0,1]))

    return inputs,lables

