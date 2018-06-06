# Trains the Model

from keras.callbacks import Callback , ModelCheckpoint , ProgbarLogger , EarlyStopping , RemoteMonitor , TensorBoard

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

class Model_trainer:

	def __init__(self , model_obj):
		self.model_obj = model_obj
		self.checkpointer = ModelCheckpoint(filepath='/Users/rishab/Desktop/fynd/solid_app/tmp/weights.hdf5', verbose=1, save_best_only=True)
		self.progbarLogger = ProgbarLogger(count_mode='steps')
		self.early_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
		self.remote_monitor = RemoteMonitor(root='http://0.0.0.0:5001', path='/publish', field='data', headers=None)
		self.history = LossHistory()
		self.board = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

	def Model_trainer_fit(self , data , one_hot_labels , epochs , batch_size ,verbose=1 , validation_data=None): #validation_data = (X_test, Y_test)

		self.model_obj.fit(data , one_hot_labels , epochs=epochs , batch_size=batch_size ,
		 verbose=verbose, validation_data=validation_data , callbacks=[self.history , self.checkpointer , self.remote_monitor , self.board])

		return self.model_obj , self.history.losses

	def Model_trainer_fit_generator(self, generator, data , target, batch_size , steps_per_epoch=None, epochs=1,
	 verbose=1, validation_data=None, validation_steps=None, class_weight=None,
	  max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0):

		self.model_obj.fit_generator(generator.flow(data , target , batch_size=batch_size),steps_per_epoch=steps_per_epoch,epochs=epochs,
		  verbose=verbose, callbacks=[self.checkpointer, self.history , self.remote_monitor], validation_data=validation_data, validation_steps=None, class_weight=None, max_queue_size=10, workers=4, use_multiprocessing=True, shuffle=True, initial_epoch=0)

		return self.model_obj , self.history.losses
