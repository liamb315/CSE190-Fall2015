import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def display_image(dataset, labels, index):
	'''Display a particular digit to screen'''
	print "Image label: ", labels[index]
	imgplot = plt.imshow(dataset[index])
	plt.show()


def preprocess_data_log(dataset, labels, k):
	''' Preprocessing code
	0.  Normalize the pixel intensities to zero mean, unit variance
	1.  Label positive class (k) as 1. All other classes are 0
	2.  Append '1' feature to dataset for intercept term'''
	X_list = []
	Y_list = []

	for i in range(0, len(dataset)):
		mean = dataset[i].mean()
		std  = dataset[i].std()
		x    = (dataset[i].flatten() - mean)/std

		X_list.append(np.append(1.0, x))

		if labels[i] == k:
			Y_list.append(1)
		else:
			Y_list.append(0)

	X = np.asarray(X_list)
	Y = np.asarray(Y_list)

	return X, Y



def preprocess_data(dataset, labels, binary_class = True):
	''' Preprocessing code
	0.  Normalize the pixel intensities to zero mean, unit variance
	1.  Extract 0 and 1 digits only from Test/Training
	2.  Append '1' feature to dataset for intercept term'''
	X_list = []
	Y_list = []

	for i in range(0, len(dataset)):
		mean = dataset[i].mean()
		std  = dataset[i].std()
		x    = (dataset[i].flatten() - mean)/std

		if binary_class == True:
			if labels[i] == 0 or labels[i] == 1:
				X_list.append(np.append(1.0, x))	
				Y_list.append(labels[i])
			
		elif binary_class == False:
			X_list.append(np.append(1.0, x))
			Y_list.append(labels[i])

	X = np.asarray(X_list)
	Y = np.asarray(Y_list)

	return X, Y

def print_performance(pred, actual):
	'''Simply takes in predicted and actual and prints performance'''
	num_incorrect = 0.0
	incorrect     = []

	for i in range(0, len(actual)):
		if pred[i] != actual[i]:
			num_incorrect += 1
			incorrect.append(i)
	#print 'Incorrect indices: ', incorrect
	print ' Performance: ', float((len(actual)-num_incorrect)/len(actual))
	print

