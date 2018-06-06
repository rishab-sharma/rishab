from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

class Add_layer:

	def __init__(self , model):
		self.model = model

	def Add_layer(self, name, input_dim=None, filters=None, kernel_size=None, border_mode = 'valid', strides=(1,1), activation=None , dropout_regularisation = 0.4):

		if name == "Convolution2D":
			self.model.add(Conv2D(filters, kernel_size = kernel_size, strides=strides, padding=border_mode, activation=activation, input_shape=input_dim))
		
		elif name == "Convolution2D_":
			self.model.add(Conv2D(filters, kernel_size = kernel_size, strides=strides, padding=border_mode, activation=activation))

		elif name == "MaxPooling2D":
			self.model.add(MaxPooling2D(pool_size=kernel_size , strides=strides , padding = 'same', data_format=None))

		elif name == "Flatten":
			self.model.add(Flatten())

		elif name == "Dense":
			self.model.add(Dense(filters , activation=activation))

		elif name == "Dropout":
			self.model.add(Dropout(dropout_regularisation))

		return self.model