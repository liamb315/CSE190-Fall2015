'''Logistic Regression for MNIST dataset'''
import numpy as np
import helper_functions as fn
import math_logistic as ml
from logisticregression import SklearnLogisticRegression
from sklearn import linear_model
import time
import matplotlib.pyplot as plt

#---------------------#
# Logistic Regression #
#---------------------#	
# Load dataset from MNIST
full_trainarray = np.load('data/numpy/trainarray.npy')
full_trainlabel = np.load('data/numpy/trainlabel.npy')
full_testarray  = np.load('data/numpy/testarray.npy' )
full_testlabel  = np.load('data/numpy/testlabel.npy' )

labels = range(0,10)
predictions = np.zeros([full_testarray.shape[0], len(labels)])

for k in labels:
	# Label positive class as 1, all other classes are 0
	X_train, Y_train = fn.preprocess_data_log(full_trainarray, full_trainlabel, k)
	X_test, Y_test   = fn.preprocess_data_log(full_testarray, full_testlabel, k)
	
	# 0.  Sklearn logistic regression
	print 'Logistic regression for label',k,'using sklearn'
	t0 = time.time()
	logreg = SklearnLogisticRegression()
	w = logreg.train(X_train, Y_train)
	t1 = time.time()
	print ' Training time:', t1-t0
	p = logreg.predict(X_test)
	probs = logreg.predict_probs(X_test)
	predictions[:,k] = probs[:,1]
	t2 = time.time()
	print ' Testing time:', t2-t1
	fn.print_performance(p, Y_test)


	'''
	# 1.  Batch gradient descent logistic regression
	print 'Logistic regression using gradient descent'
	t0 = time.time()
	w = np.zeros(X_train.shape[1])
	w = ml.gradient_descent(X_train, Y_train, w, 35)
	t1 = time.time()
	print ' Training time:', t1-t0
	p = ml.predict_logistic(X_test, w)
	t2 = time.time()
	print ' Testing time:', t2-t1
	fn.print_performance(p, Y_test)


	# 2.  Stochastic gradient descent logistic regression
	print 'Logistic regression using stochastic gradient descent'
	t0 = time.time()
	w = np.zeros(X_train.shape[1])
	w = ml.stochastic_gradient_descent(X_train, Y_train, w, 50, 256)
	t1 = time.time()
	print ' Training time:', t1-t0
	p = ml.predict_logistic(X_test, w)
	t2 = time.time()
	print ' Testing time:', t2-t1
	fn.print_performance(p, Y_test)
	'''

log_predictions = np.argmax(predictions, axis=1)
fn.print_performance(log_predictions, full_testlabel)
