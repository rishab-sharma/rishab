from ..model_builder.add_layer import add_layer
from ..model_builder.model_init import model_init
from ..model_builder.model_trainer import model_trainer
from ..model_builder.model_compiler import model_compiler
from ..utils.data_gen import data_gen
from ..utils.face_crop import face_crop
from ..utils.resize import resize

class Parser:

	def __init__(self,data):
		self.data = data

	def Parser_layers(self , option="model_config" , option_="layers"):
		layers = self.data[option][option_]
		for layer in layers:
			seq = layer["seq"]
			nmae = layer["name"]
			param = layer["param"]
			filters = param["filters"]
			kernel = param["kernel"]
			activation = param["activation"]
			stride = param["stride"]
			dropout_regularisation = param["dropout_regularisation"]
		pass

	def Parser_preprocess(self , option="preprocess"):
		preprocess = self.data[option]
		choice = preprocess["choice"]
		keras_ = preprocess["keras"]
		custom = preprocess["custom"]
		rotation_range = keras_["rotation_range"]
		width_shift_range = keras_["width_shift_range"]	
		height_shift_range = keras_[height_shift_range]
		horizontal_flip = keras_["horizontal_flip"]
		vertical_flip = keras_["vertical_flip"]
		flip_top_bottom = custom["flip_top_bottom"]
		rotate = custom["rotate"]
		random_distortion = custom["random_distortion"]
		flip_left_right = custom["flip_left_right"]
		pass

	def Parser_model_config(self , option="model_config"):
		model_config = self.data[option]
		type_ = model_config["type"]
		compiler = model_config["compiling_setting"]
		input_dim = compiler["input_dim"]
		loss_function = compiler["loss_function"]
		optimiser = compiler["optimiser"]
		learning_rate = compiler["learning_rate"]
		callbacks = compiler["callbacks"]
		pass

	def Parser_fine_tune(self , option="fine_tune"):
		fine_tune = self.data[option]
		phases = fine_tune["phases"]
		base_trainable_layers = fine_tune["base_trainable_layers"]
		pass

