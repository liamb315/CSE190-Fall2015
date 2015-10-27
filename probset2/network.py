import numpy as np
import random as r




class network:
	'''Class for a neural network'''	
	def __init__(self, topology):
		self.topology   = topology
		self.num_layers = len(topology)
		self.weights    = [np.random.randn(y,x) for x,y in zip(topology[:-1], topology[1:])] 

	def forward(self, a):
		'''Forward propagation'''
		for w in zip(self.weights):
			a = sigmoid(np.dot(w, a))
		return a

	def backward(self):
		'''Backward propagation'''
		pass

	def loss(self):
		'''Loss for the network '''
		pass

	def get_parameters(self):
		return self.weights



def sigmoid(a):
	return 1.0/(1.0 + np.exp(-a))

