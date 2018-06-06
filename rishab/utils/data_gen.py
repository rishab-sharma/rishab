from keras.preprocessing.image import ImageDataGenerator

class Data_gen:

	def __init__(self , X_train , y_train):
		self.X_train = X_train
		self.y_train = y_train

	def Data_gen(self , featurewise_center=False , samplewise_center=False , featurewise_std_normalization=False , 
samplewise_std_normalization=False, zca_whitening=False , rotation_range=40 , width_shift_range=0.2 ,
height_shift_range=0.2 , horizontal_flip=True , vertical_flip=False):

		print(" * Generating Image Data > using default Keras Modules *")
		
		datagen = ImageDataGenerator(
		    featurewise_center=featurewise_center,
		    samplewise_center=samplewise_center, 
		    featurewise_std_normalization=featurewise_std_normalization, 
		    samplewise_std_normalization=samplewise_std_normalization, 
		    zca_whitening=zca_whitening, 
		    rotation_range=rotation_range,  
		    width_shift_range=width_shift_range, 
		    height_shift_range=height_shift_range, 
		    horizontal_flip=horizontal_flip,
		    vertical_flip=vertical_flip)

		datagen.fit(self.X_train)
		return datagen

# Returns the datagenerator object