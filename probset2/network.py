import numpy as np
import random as r




class network:
	'''Class for a neural network'''	
	def __init__(self, topology):
		self.topology   = topology
		self.num_layers = len(topology)
		self.weights    = [np.random.randn(y,x) for x,y in zip(topology[:-1], topology[1:])] 
		self.a          = [np.zeros(l) for l in topology]
		self.z          = [np.zeros(l) for l in topology]
		

	def forward(self, x):
		'''Forward propagation'''
		self.a[0] = x
		self.z[0] = x
		for l in range(0, len(self.weights)):
			self.a[l+1] = np.dot(self.weights[l], x)
			x = sigmoid(np.dot(self.weights[l], x))
			self.z[l+1] = x
		return x
	

	def backward(self, x, t):
		'''Backward propagation'''
		grad_w      = [np.zeros(w.shape) for w in self.weights]
		delta       = [np.zeros(l) for l in self.topology]
		N           = self.num_layers

		# Compute delta for output layer
		delta[N-1]  = self.z[N-1] - t
		grad_w[N-2] = np.dot(delta[N-1], self.z[N-2].T)
		
		# Compute delta for hidden layers
		for l in range(N-2, -1, -1):
			delta[l]    = np.dot(self.weights[l].T, delta[l+1]) * sigmoid_derivative(self.a[l])
			grad_w[l-1] = np.dot(delta[l], self.z[l-1].T)
		return grad_w


	def error(self):
		'''Loss for the network '''
		pass


	def error_derivative(self):
		pass



def sigmoid(a):
	return 1.0/(1.0 + np.exp(-a))

def sigmoid_derivative(a):
	return sigmoid(a) * (1.0 - sigmoid(a))
