import numpy as np
import network as nn

# Data example
x = np.array([[1,1]]).T
t = 1.0

# Establish a network
net = nn.network([2,3,5,1])


print net.forward(x)
grad_w, grad_b = net.backward(x, t)


for i in range(0, len(net.weights)):
	net.weights[i] = net.weights[i] - 1.0 * grad_w[i]
	net.biases[i]  = net.biases[i] - 1.0 * grad_b[i]

print net.forward(x)
grad_w, grad_b = net.backward(x, t)

for i in range(0, len(net.weights)):
	net.weights[i] = net.weights[i] - 1.0 * grad_w[i]
	net.biases[i]  = net.biases[i] - 1.0 * grad_b[i]

print net.forward(x)
