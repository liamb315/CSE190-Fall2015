import numpy as np
import random as r




class network:
'''Class for a neural network'''	
	def __init__(self, topology):
		self.num_layers = len(topology)
		self.weights    = [np.random.randn(y,x) for x,y in zip(topology[:-1], topology[1:])] 

	def forward(self, input):
		'''Forward propagation'''
		pass

	def backward(self):
		'''Backward propagation'''
		pass

	def loss(self):
		'''Loss for the network '''
		pass

