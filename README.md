# LSTM from scratch in NumPy
This is an LSTM that I made from scratch in NumPy without high level libraries. I made this to understand how graidients flow accross timesteps and imporve my confidance working with NumPy and vector calculus

**What's included?:**

big_modle_handleing.py - linkins the LSTM and the FFN together. the FFN takes the hiddern state from the LSTM at each timesteap as input and generates the output.

chad_LSTM.py - The core LSTM. Contains forwards pass, backwars pass and initilisation.

running_parity_dataset.py - generates a partiy dataset with one-hot lables

test.py - performs numerical graidient testing

time_ffn.py - an FFN made to handle inputs with a timestep dimention. It treats the timestep dimention as another batch dimention and doesnt flatten the inptus.

**requierments**
NumPy
matplotlib, (for graphing cost)

**implementation overveiw**
The LSTM is mathmaticaly stadard in the forwards and backwards pass. It used forget, candidate, input and output gates.

The FFN takes input of shape (seq_len,batch_size,fetures) It uses numpy brordcasting to proccess the data without flattening it, preserving the timestep dimention

To solve parity I used a dynamic curriculum, where i incrice the sequance length when cost falls below 0.2. I chose 0.2 becasue this is the value the cost plaued around in training
