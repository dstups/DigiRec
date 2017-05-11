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

##Takes a list of categories and returns a list of one hot encoded tuples where the first element is the 
#Category and the second element is the one-hot encoding for that element

def to_onehot(List):
	final_list = []
	for i in List:
		position = List.index(i)
		dummy_list = [0] * len(List)
		dummy_list[position] = 1
		final_list.append((i, dummy_list))
	return final_list
	
#print to_onehot(["This", "that", "the other"])	







