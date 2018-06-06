import cv2

"""
OpenCV file download
!wget https://sourceforge.net/projects/opencvlibrary/files/opencv-unix/3.4.1/opencv-3.4.1.zip
"""
class Face_crop:
	def __init__(self , image):
		self.image = image
		
	def Face_crop_haar(self , address):
		print("* Attempting to crop out a face using Haar Cascades")
		fc = cv2.CascadeClassifier(address)
		img = cv2.imread(self.image)
		gray =cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
		faces = fc.detectMultiScale(gray , 1.3 , 5)
		if len(faces) == 0:
			return img
		else:
			for (x,y,w,h) in faces:
				crop_img = img[y+h+10:y+(h*4), x-w+10:x+(w*2)-10]
			return crop_img

# Returns the cropped Image 
