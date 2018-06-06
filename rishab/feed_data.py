import os
import numpy as np
from multiprocessing import Pool
import time
from utils.map_reduce import map_reduce
from sklearn.model_selection import train_test_split

class Feed_data:
	def __init__(self,dir_name):
		self.train_dir = dir_name

	def Feed_data(self):
		p = Pool()
		cwd = os.getcwd()
		path = cwd + '/data/'
		self.train_dir = 'sample_data'
		class_names = os.listdir(path + self.train_dir)
		t1 = time.time()
		a = p.map(map_reduce,class_names)
		print("Time Taken by Map Reduce to process the Data: ",time.time() - t1)
		X = []
		y = []
		for i in a:
			X += i[0]
			y += i[1]

		X = np.array(X)
		y = np.array(y)

		y = np.eye(len(np.unique(y)))[y]

		X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 3)

		return X_train , X_test , y_train , y_test