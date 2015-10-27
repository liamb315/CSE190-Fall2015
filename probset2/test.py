import numpy as np
import network as nn


x = np.array([[1,1]]).T
net = nn.network([2,3,1])
print net.get_parameters()

print net.forward(x)