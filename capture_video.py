# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import os
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.sharpness = 50
rawCapture = PiRGBArray(camera, size=(640, 480))
 
face_cascade = cv2.CascadeClassifier('/home/pi/mainak/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_default.xml')
FACE_COUNT = 50
filename = 'store_faces_debjit.npy'

# allow the camera to warmup
time.sleep(0.1)

face_images = []
facecount = 0
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array

	im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(im_gray, 1.3, 5)
	if(len(faces) > 0):
		print "# faces: ", len(faces)
	for (x,y,w,h) in faces:
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		face_gray = np.array(im_gray[y:y+h, x:x+w], 'uint8')
		face_sized = cv2.resize(face_gray, (30, 30))
		face_images.append(face_sized)
		facecount += 1

	#     #roi_gray = im_gray[y:y+h, x:x+w]
	#     #roi_color = img[y:y+h, x:x+w]
 
	# show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
	#cv2.waitKey(1)
	#cv2.destroyAllWindows()
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	#print "Image: ", image.shape
 
	# if the `q` key was pressed, break from the loop
	# if key == ord("q"):
	if (facecount == FACE_COUNT) or (key == ord("q")):
		break

#np.save(filename, face_images)
#face_images_np = np.copy(face_images)
face_images_np = np.asarray(face_images, dtype = 'uint8')
if (os.path.exists(filename)):
	#print " abc"
	np.save(filename, np.concatenate((np.load(filename), face_images_np), axis=0))
else:
	#print "xyz"
	np.save(filename, face_images_np)
