**LSTM**

This is an LSTM that I made from scratch in NumPy without high level libraries. I made this to understand how gradients flow across timesteps and improve my confidence working with NumPy and vector calculus.

**What's included?:**

big_model_handling.py - links the LSTM and the FFN together. The FFN takes the hidden state from the LSTM at each timestep as input and generates the output.

chad_LSTM.py - The core LSTM. Contains forward pass, backward pass and initialisation.

running_parity_dataset.py - generates a parity dataset with one-hot labels

test.py - performs numerical gradient testing.

time_ffn.py - an FFN made to handle inputs with a timestep dimension. It treats the timestep dimension as another batch dimension and doesn't flatten the inputs.

**how to run**

python big_model_handling.py - trains and tests on parity

python test.py - tests against numerical gradients

**requirements**

NumPy

matplotlib (for graphing loss)

**implementation overview**

The LSTM is mathematically standard. It uses forget, candidate, input and output gates. My implementation uses timestep indexing such that at timestep t the input hiddern state is h[t] and the ouput hiddern state is h[t+1]. 

The FFN takes input of shape (seq_len, batch_size, features). It uses numpy broadcasting to process the data without flattening it, preserving the timestep dimension.

To solve parity I used a dynamic curriculum, where I increase the sequence length when cost falls below 0.2. I chose 0.2 because this is the value the cost played around in training.

Starting on small sequence lengths works better because after the network 'loses track' of the parity outputs lose any relation to the labels. This means gradients from beyond the point of first failure become noise. For this reson it is better to begin by training on small sequence lengths and scaling up to keep any failure points close to the end of the sequence, minimiseing the gradients that become noise, making training more efficient. This method allows the network to achieve 100% accuracy on a sequence length of 1000 across 10,000 test examples.
