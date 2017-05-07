import pandas as pd 
import numpy as np 
import tensorflow as tf 


df_train = pd.read_csv('data/train.csv')
arr_train = df_train.as_matrix(columns=None)

print (arr_train.shape)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros[784, 10])
y = tf.Variable(tf.zeros[10])
