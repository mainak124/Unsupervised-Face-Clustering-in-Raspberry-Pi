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

# allow the camera to warmup
time.sleep(0.1)

def capture_and_detect():
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image, face_images = capture_and_detect(frame)
     
    	# show the frame
    	cv2.imshow("Frame", image)
    	key = cv2.waitKey(1) & 0xFF
    	#cv2.waitKey(1)
    	#cv2.destroyAllWindows()
     
    	# clear the stream in preparation for the next frame
    	rawCapture.truncate(0)
     
    	# if the `q` key was pressed, break from the loop
    	if key == ord("q"):
    		break

def capture_and_detect(frame):
	image = frame.array
	im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(im_gray, 1.3, 5)
    face_images = []
	for (x,y,w,h) in faces:
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		face_gray = np.array(im_gray[y:y+h, x:x+w], 'uint8')
		face_sized = cv2.resize(face_gray, (30, 30))
    	face_images.append(face_sized)
    return image, face_images
