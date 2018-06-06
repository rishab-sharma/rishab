import os
import time
import numpy as np
import cv2
from tqdm import tqdm

t1 = time.time()

cwd = os.getcwd()

mypath = cwd + '/data/'
train_dir = 'sample_data/'

class_names = os.listdir(mypath + train_dir)

n_class = len(class_names)

X=[]
y=[]

c = 0

for i in class_names:
    con = os.listdir(mypath + train_dir + str(i))
    for j in tqdm(con):
        image  = cv2.imread(mypath + train_dir + str(i) + '/' + j)
        im = cv2.resize( image , (299 , 299 ), interpolation = cv2.INTER_CUBIC)
        X.append(im)
        y.append(c)
    c+=1

print(time.time() - t1)