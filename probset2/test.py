import numpy as np
import network as nn

# Data example
x = np.array([[1,1]]).T
t = 1.0

# Establish a network
net = nn.network([2,3,1])

#print net.a
#print net.z
print net.forward(x)

grad_w = net.backward(x, t)

for i in range(0, len(net.weights)):
	net.weights[i] = net.weights[i] - 0.01 * grad_w[i]

print
print net.forward(x)

