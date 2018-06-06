import cv2

class Resize:

	def __init__(self , image):
		self.image = image

	def Resize(self , x , y): #Pass the image and x , y dimensions to be resized to.

		# print("* Resizing Image to {} x {} Pixels".format(x,y))
		
		img = cv2.resize( self.image , (x , y ), interpolation = cv2.INTER_CUBIC)
	
		return img

# Returns the resied image