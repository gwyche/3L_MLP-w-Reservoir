# 3L_MLP-w-Reservoir
3 Layer MLP NN attached to a reservoir.

To try, just run the python script. The weight matrices and input vector will be assigned random numbers.

The NN class is called from the commands at the bottom of the program:
nn = StandardNN(30,.023,.0000001,2000,.9,True)
nn.train()

You're free to vary each of the hyperparameters above. 
They are respectively: layerSizeArg, learningRateArg, biasArg, runsArg, reservoirSparsenessArg, bypassReservoirArg.
layerSizeArg is the width of the MLP and Reservoir. MLP layer sizes will be made independent of one another in a later iteration.
learningRateArg is the learning rate.
biasArg is the bias.
runsArg is how many times the NN will feedforward and backpropagate.
reservoirSparsenessArg is the sparseness of the reservoir.
bypassReservoirArg is a True or False and determines whether to use the reservoir or to run strictly as an MLP

Input size is fixed at 10 and is embedded in the higher dimensional layer sizes.
At the moment, the input vector is fixed at 10.
The target vector is fixed as well.
In later versions that process an actual data set, the input and output vectors will of course update in realtime.

You will need PyTorch to run it. I'm using version 2.5.1 and Python version 3.12.2.

