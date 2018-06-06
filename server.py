"""
Creator: Rishab Sharma
contact: rishabsharmaddn@gmail.com
Cheers!
"""
from flask import Flask , render_template , request
from utils.face_crop import Face_crop
from utils.resize import Resize
from utils.augmentor import Augmentor_class
from utils.data_gen import Data_gen
from model_builder.model_init import Model_init
from model_builder.add_layer import Add_layer
from model_builder.model_compiler import Model_compiler
from model_builder.model_trainer import Model_trainer
from keras.datasets import mnist
import cv2
from keras.utils import to_categorical
import json
import multiprocessing as mp
import logging as log
from feed_data import Feed_data
from multiprocessing import Pool
import time
import os

"""
	Logger:
	# Debug : Detailed Information
	# Info : Confirmation , everything is going Good
	# Warning : Something Unexpected happen 
	# Error : Some Functions not Executed
	# Critical : Program may stop
"""

log.basicConfig(filename='logs/training.log', level=log.DEBUG, format='%(asctime)s:%(levelname)s:%(process)d:%(processName)s:%(message)s')

app = Flask(__name__)

@app.route('/',methods=["GET" , "POST"])
def index():
	return "Auto ML Initiated"


@app.route('/doit',methods=["GET" , "POST"])
def doit():
	data = request.get_json()

	########## Reading the Data ##############

	train_address = data['train_address']
	data_obj = Feed_data(train_address)
	X_train , X_test , y_train , y_test = data_obj.Feed_data()
	
	########### Creating the Generator / Preprocessing #############
	
	generator_obj = Data_gen(X_train,y_train)
	generator = generator_obj.Data_gen()

	########### Assembling the Model #############

	model_obj = Model_init("InceptionV3")
	model = model_obj.Model_init()
	print(model.summary())
	
	########### Saving the Model #############

	model_json = model.to_json()
	with open("jsons/model.json", "w") as json_file:
	    json_file.write(model_json)
	print(" * Model Saved in disk as JSON * ")
	
	########### Compiling the Model #############

	compiler_obj = Model_compiler(model)
	model = compiler_obj.Model_compiler(optimizer='adam' , loss='categorical_crossentropy')
	print(model)
	
	############ Training the model ############

	fit_obj = Model_trainer(model)
	model , history_losses = fit_obj.Model_trainer_fit_generator( generator=generator, data=X_train , target=y_train, batch_size=30, steps_per_epoch=120,validation_data = (X_test,y_test))
	
	########### Saving the Trained Model #############

	model.save('models/model.h5')
	print(history_losses)
	
	########################
	
	return "* Model Created and Saved in your Disk *"

@app.route('/publish',methods=["GET","POST"])
def publish():
	print(" * Publishing the Logs *")
	data = request.form
	data = json.loads(data['data'])
	log.info("* Epoch => " + str(data['epoch']))
	log.info("* Accuracy => " + str(data['acc']))
	log.info("* Loss => " + str(data['loss']))
	log.info("* Val_Accuracy => " + str(data['val_acc']))
	log.info("* Val_loss => " + str(data['val_loss']))
	return "* Published *"

def server1(port):
	app.run(host='0.0.0.0',debug=True , port=port)

def server2(port):
	app.run(host='0.0.0.0',debug=True , port=port)

if __name__ == "__main__":
	print(" * Creating Multiple processes of the server on ports 5000 and 5001 *")
	p1 = mp.Process(target=server1 , args=(5000,))
	p2 = mp.Process(target=server2 , args=(5001,))
	log.info(" * Evaluator Initiated *")
	log.info(" * Publisher Initiated *")
	p1.start()
	p2.start()
	p1.join()
	p2.join()
