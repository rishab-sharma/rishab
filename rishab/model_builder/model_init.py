from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

class Model_init:

	def __init__(self , model_type):
		self.model_type = model_type

	def Model_init_custom(self):
		
		model = Sequential()

		print("* Initialising a {} Model".format(self.model_type))

		return model

	def Model_init(self):

		base_model = InceptionV3(weights='imagenet', include_top=False)
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(12, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
		
		for layer in base_model.layers:
		    layer.trainable = False


		return model