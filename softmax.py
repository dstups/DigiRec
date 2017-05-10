import pandas as pd 
import numpy as np 
import tensorflow as tf 
from random import randint
from tensorflow.python import debug as tf_debug
from dataproc import to_array, randomsample


arr_train = to_array('data/train.csv')
arr_test = to_array('data/test.csv')

print (arr_train.shape)

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
	sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
	tf.global_variables_initializer().run()

	for i in range(1000):
		start_row, end_row = randomsample(arr_train, 100)
		x_train = arr_train[start_row:end_row, 1:]
		y_train = arr_train[start_row:end_row, 0]
		sess.run(train_step, feed_dict={x: x_train, y_: y_train})


	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	x_test = arr_test[:, 1:]
	y_test = arr_test[:, 0]

	print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))