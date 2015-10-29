import numpy as np
import network as nn
import optimization as opt


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Load dataset from MNIST
x_train = np.load('data/numpy/trainarray.npy')
t_train = np.load('data/numpy/trainlabel.npy')
x_test  = np.load('data/numpy/testarray.npy' )
t_test  = np.load('data/numpy/testlabel.npy' )

x_train = [np.ravel(x) for x in x_train]
x_train = np.asarray(x_train)

t_train = [vectorized_result(t) for t in t_train]
t_train = np.asarray(t_train)
t_train = np.squeeze(t_train)

x_test  = [np.ravel(x) for x in x_test]
x_test  = np.asarray(x_test)

t_test  = [vectorized_result(t) for t in t_test]
t_test  = np.asarray(t_test)
t_test  = np.squeeze(t_test)

# Training options
options = {'learn_rate':0.01, 'learn_rate_dec': 0.9, 'batch_size': 256, 'epochs': 10, 'criterion': 'cross_entropy'}

# Fully-connected neural network
topology = [784, 256, 10]
net = nn.network(topology)

# Train model 
trainer = opt.optimization(net, options)
trainer.stochastic_grad_descent(x_train, t_train)

# Test model




