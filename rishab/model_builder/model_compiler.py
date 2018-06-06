# Compiles The Model

class Model_compiler:

	def __init__(self , model_obj):
		self.model_obj = model_obj

	def Model_compiler(self , optimizer , loss , metrics=['accuracy']):

		self.model_obj.compile(optimizer=optimizer,loss=loss,metrics=metrics)

		return self.model_obj