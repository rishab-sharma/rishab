from multiprocessing import Pool
from resize import Resize
import os
import cv2
import time


def map_reduce(class_names):
	train_dir = 'sample_data/'
	X,y = [],[]
	cwd = os.getcwd()
	path = cwd + '/data/'
	hold = os.listdir(path + train_dir)
	con = os.listdir(path + train_dir + class_names)
	print(len(con),class_names)
	dict_={}
	for i,a in enumerate(hold):
		dict_[a] = i
	for j in con[:20]:
		image  = cv2.imread(path + train_dir + class_names + '/' + j)
		shape = Resize(image)
		im = shape.Resize(299 , 299)
		X.append(im)
		y.append(dict_[class_names])
	return [X,y]

if __name__ == "__main__":

	p = Pool()
	cwd = os.getcwd()
	path = cwd + '/data/'
	train_dir = 'sample_data'
	class_names = os.listdir(path + train_dir)
	t1 = time.time()
	a = p.map(map_reduce,class_names)
	print(a)
	print(time.time() - t1)