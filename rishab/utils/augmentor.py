import Augmentor

class Augmentor_class:

	def __init__(self , address):
		self.address = address

	def __Augmentor__(self , batch_size , p_ftb=0.1 , p_r=0.1 , mlr=5 , mrr=5 ,  p_rd=0.1 ,
 p_flr=0.1 , p_r90=0.1 , p_r270=0.1 , p_cr=0.1 , percent_area=0.5, p_resize=0.1 ):
	
		print("* Creating Augmentation Pipeline")

		p = Augmentor.Pipeline(self.address)

		p.flip_top_bottom(probability=p_ftb)
		p.rotate(probability=p_r, max_left_rotation=mlr, max_right_rotation=mrr)
		p.random_distortion(probability=p_rd, grid_width=4, grid_height=4, magnitude=8)
		p.flip_left_right(probability=p_flr)
		p.rotate90(probability=p_r90)
		p.rotate270(probability=p_r270)
		p.crop_random(probability=p_cr, percentage_area=percent_area)
		p.resize(probability=p_resize, width=120, height=120)

		# p.sample(100)
		# p.status()
		g = p.keras_generator(batch_size=batch_size)

		steps_per_epoch=len(p.augmentor_images)/batch_size

		print()
		print("* Data Augumented and generator Created")

		return g, steps_per_epoch

# Returns the Keras fit_generator() generator and steps per epoch parameter