import pandas as pd 
import numpy as np 
import tensorflow as tf 
from random import randint
from tensorflow.python import debug as tf_debug
from dataproc import to_array, randomsample, to_onehot
from tensorflow.examples.tutorials.mnist import input_data


arr_train = to_array('data/train.csv')
arr_test = to_array('data/test.csv')

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# tensor to hold cross-entropy results
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
 
# create the session to run the graph
with tf.Session() as sess:
	# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
	tf.global_variables_initializer().run()

	for i in range(1000):
		start_row, end_row = randomsample(arr_train, 100)
		# first column of arr_train is labels so we exlude that
		x_train = arr_train[start_row:end_row, 1:] 
		# want only the first column - i.e labels 0-9 so we can encode them as one-hots
		y_raw = arr_train[start_row:end_row, 0]
		y_train = to_onehot(y_raw)
		sess.run(train_step, feed_dict={x: x_train, y_: y_train})

	# Think the error is here because the training step appears to be running ok
	# y_test results in a matrix of nans
	x_test = arr_test[:, 0:]
	feed_dict = {x: x_test}
	y_test = sess.run(y, feed_dict)
	print (y_test)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print ("Accuracy: %.2f " %( float(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))))
