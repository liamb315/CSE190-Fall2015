import numpy as np
import random as r

class optimization:
	'''Class for optimizing a network '''
	def __init__(self, network, opt):
		self.network        = network
		self.learn_rate     = opt['learn_rate']
		self.learn_rate_dec = opt['learn_rate_dec']
		self.learn_rate_dec = opt['learn_rate_dec']
		self.batch_size     = opt['batch_size']
		self.epochs         = opt['epochs']

	def stochastic_grad_descent(self, x, t):
		'''Performs stochastic gradient descent on self.network'''
		indices = r.sample(xrange(len(x)), self.batch_size)
		x_batch = x[indices]
		t_batch = t[indices]

		eta    = self.learn_rate 
		W      = [np.zeros(w.shape) for w in self.network.weights]
		b      = [np.zeros(b.shape) for b in self.network.biases ]

		for i in range(0, self.epochs):
			grad_W, grad_b = self.calculate_gradient_batch(x_batch, t_batch)

			W = W - eta * grad_W
			b = b - eta * grad_b

			eta = eta/(1.0 + i * learn_rate_decay)

		self.network.update_weights(W)
		self.network.update_biases(b)

	def calculate_gradient_batch(self, x, t):
		'''Calculate the gradients for a random set of examples of size self.batch_size'''
		weights = self.network.weights
		biases  = self.network.biases

		grad_w = [np.zeros(w.shape) for w in weights]
		grad_b = [np.zeros(b.shape) for b in biases]

		for i in range(0, len(x)):
			_ = self.network.forward(x[i]) # Forward prop for activations
			delta_grad_w, delta_grad_b = self.network.backward(x[i], t[i]) # Backward prop for gradients

			grad_w = [g_w + d_gw for g_w, d_gw in zip(grad_w, delta_grad_w)]
			grad_b = [g_b + d_gb for g_b, d_gb in zip(grad_b, delta_grad_b)]

		return grad_w, grad_b

	def calculate_cross_entropy_loss(self, y, t):
		'''Calculate loss'''
		return np.sum(np.nan_to_num(-y*np.log(y)) - (1-y)*np.log(1-y))

