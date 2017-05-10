import pandas as pd 
import numpy as np 
from random import randint


def to_array(filepath):
	df_train = pd.read_csv(filepath)
	arr_train = df_train.as_matrix(columns=None)
	return arr_train


def randomsample(array, interval):
	max_index = len(array) - (1  + interval)
	random_index = randint(0, max_index)

	start_row = random_index
	end_row = start_row + interval
	return start_row, end_row







