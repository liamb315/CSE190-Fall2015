import numpy as np
import network as nn
import optimization as opt

# Load dataset from MNIST
x_train = np.load('data/numpy/trainarray.npy')
t_train = np.load('data/numpy/trainlabel.npy')
x_test  = np.load('data/numpy/testarray.npy' )
t_test  = np.load('data/numpy/testlabel.npy' )

# Training options
options = {'learn_rate':0.01, 'learn_rate_dec': 0.9, 'batch_size': 256, 'epochs': 10, 'criterion': 'cross_entropy'}

# Fully-connected neural network
topology = [784, 256, 10]
net = nn.network(topology)

# Train model 
trainer = opt.optimization(net, options)
trainer.stochastic_grad_descent(x_train, t_train)

# Test model




