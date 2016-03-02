import numpy as np
import cPickle as pkl
from stacked_autoencoder import SdA
import theano
import cv2

N_DIM = 100
IS_KMEANS = 1

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.sharpness = 50
rawCapture = PiRGBArray(camera, size=(640, 480))
 
face_cascade = cv2.CascadeClassifier('/home/pi/mainak/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_default.xml')

# allow the camera to warmup
time.sleep(0.1)

def label_faces_from_video(centers):
    # loading the trained model
    model_file = file('models/pretrained_model.save', 'rb')
    sda = pkl.load(model_file)
    model_file.close()
    
    get_single_encoded_data = sda.single_encoder_function()

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image, face_images = capture_and_detect(frame)
        for face in face_images:
            encoded_x = get_single_encoded_data(train_x=face)
            if (IS_KMEANS == 1):
                label_x = get_kmeans_labels(centers, encoded_x)
            # TODO else:    
            # TODO     label_x = get_tseries_labels(encoded_x)
            print("This is person: ", label_x)
     
    	# show the frame
    	cv2.imshow("Frame", image)
    	key = cv2.waitKey(1) & 0xFF
     
    	# clear the stream in preparation for the next frame
    	rawCapture.truncate(0)
     
    	# if the `q` key was pressed, break from the loop
    	if key == ord("q"):
    		break

def get_kmeans_labels(centers, x):
    dist = []
    for center in centers:
        dist.append(np.linalg.norm(center-x))
    return np.argmin(np.asarray(dist))

# TODO def get_tseries_labels(x):

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

def cluster_train_data():

    train_set_x = theano.shared(value = np.load('new_data/train_faces.npy'), borrow=True)
    test_set_x  = theano.shared(value = np.load('new_data/test_faces.npy'), borrow=True)
    
    # compute number of minibatches for training, validation and testing
    n_data = train_set_x.get_value(borrow=True).shape[0]
    train_x = np.zeros((n_data, N_DIM), dtype=np.float32)
    
    # loading the trained model
    model_file = file('models/pretrained_model.save', 'rb')
    sda = pkl.load(model_file)
    model_file.close()
    
    get_encoded_data = sda.encoder_function(train_set_x=train_set_x)

    for i in range(n_data):
        encoded_x = get_encoded_data(index=i)
        if (IS_KMEANS == 1):
            train_x[i] = encoded_x
        # TODO else:


    if (IS_KMEANS == 1):

        #flags = cv2.KMEANS_RANDOM_CENTERS
        flags = cv2.KMEANS_PP_CENTERS
        # Apply KMeans
        compactness, labels, centers = cv2.kmeans(data=train_x, K=3, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100000, 0.001), attempts=10, flags=flags)
        print "Error: ", compactness, labels
        A = []
        B = []
        C = []
        # Now split the data depending on their labels
        for i in labels:
            if (labels[i] == 0):
                A.append(train_x[i])
            elif (labels[i] == 1):
                B.append(train_x[i])
            elif (labels[i] == 2):
                C.append(train_x[i])
        print "Length: ", len(A), len(B), len(C)
    # TODO else:

    return centers


if __name__ == '__main__':
    centers = cluster_train_data()
