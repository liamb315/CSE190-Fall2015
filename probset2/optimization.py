import numpy as np
import random as r

class optimization:
	'''Class for optimizing a network '''
	def __init__(self, network, opt):
		self.network        = network
		self.learn_rate     = opt['learn_rate']
		self.learn_rate_dec = opt['learn_rate_dec']
		self.batch_size     = opt['batch_size']
		self.epochs         = opt['epochs']

	def stochastic_grad_descent(self, x, t):
		'''Performs stochastic gradient descent on self.network'''
		assert len(x) == len(t)
		eta    = self.learn_rate 
		
		for i in xrange(self.epochs):
			print 'Epoch', i 

			eta  = eta/(1.0 + i * self.learn_rate_dec)
			x_rand, t_rand = shuffle_in_place(x, t)
			N  = int(len(x)/self.batch_size)*self.batch_size  # Ensure batches divide evenly
			
			for j in xrange(0, N, self.batch_size):
				W      = [np.zeros(w.shape) for w in self.network.weights]
				b      = [np.zeros(b.shape) for b in self.network.biases ]

				x_batch = x_rand[j:j+self.batch_size]
				t_batch = t_rand[j:j+self.batch_size]

				grad_W, grad_b = self.calculate_gradient_batch(x_batch, t_batch)

				W = [(w - eta * g_w) for w, g_w in zip(W, grad_W)]
				b = [(b - eta * g_b) for b, g_b in zip(b, grad_b)]

				self.network.update_weights(W)
				self.network.update_biases(b)

				print ' batch accuracy', self.network.network_accuracy(x_batch, t_batch)
			
			

	def calculate_gradient_batch(self, x, t):
		'''Calculate the gradients for a set of examples'''
		assert len(x) == len(t)

		weights = self.network.weights
		biases  = self.network.biases

		grad_w = [np.zeros(w.shape) for w in weights]
		grad_b = [np.zeros(b.shape) for b in biases]

		for i in xrange(len(x)):
			_ = self.network.forward(x[i]) # Forward prop for activations
			delta_grad_w, delta_grad_b = self.network.backward(x[i], t[i]) # Backward prop for gradients

			grad_w = [g_w + d_gw for g_w, d_gw in zip(grad_w, delta_grad_w)]
			grad_b = [g_b + d_gb for g_b, d_gb in zip(grad_b, delta_grad_b)]

		return grad_w, grad_b

	def cross_entropy_loss(self, y, t):
		'''Calculate loss for output y and target t'''
		return np.sum(np.nan_to_num(-y*np.log(y)) - (1-y)*np.log(1-y))

def shuffle_in_place(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


