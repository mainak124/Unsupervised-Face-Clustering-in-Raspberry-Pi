from __future__ import division
import numpy as np
import cPickle as pkl
from stacked_autoencoder import SdA
import theano
import cv2
import io
import math

N_DIM = 100
PARTITION = 10
IS_KMEANS = 1
train_mean = np.load('new_data/store_mean.npy')

def label_faces_from_video(centers):

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    camera.sharpness = 50
    rawCapture = PiRGBArray(camera, size=(640, 480))
     
    face_cascade = cv2.CascadeClassifier('/home/pi/mainak/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_default.xml')
    
    # allow the camera to warmup
    time.sleep(0.1)

    # loading the trained model
    model_file = file('models/pretrained_model.save', 'rb')
    sda = pkl.load(model_file)
    model_file.close()
    
    get_single_encoded_data = sda.single_encoder_function()

    time = 1
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image, face_images = capture_and_detect(frame)
        
        for face in face_images:
            encoded_x = get_single_encoded_data(train_x=face)
            if (IS_KMEANS == 1):
                label_x = get_kmeans_labels(centers, encoded_x)
            # else:    
            #     label_x = cluster.get_tseries_labels(encoded_x,time)
            print("This is person: ", label_x)
        time += 1
            

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

def capture_and_detect(frame):
    image = frame.array
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im_gray, 1.3, 5)
    face_images = []
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        face_gray = np.array(im_gray[y:y+h, x:x+w], 'uint8')
        face_sized = cv2.resize(face_gray, (30, 30))

        flat_face = face_sized.reshape(1, face_sized.shape[0]*face_sized.shape[1])
        flat_face = flat_face/255
        face_x = flat_face - train_mean

        face_images.append(face_x)
    return image, face_images

def cluster_train_data():
    
    train_set = np.load('new_data/train_faces.npy')
    test_set = np.load('new_data/test_faces.npy')

    tr_x = [i[0] for i in train_set]
    tr_y = [i[1] for i in train_set]
    te_x = [i[0] for i in test_set]
    te_y = [i[1] for i in test_set]
    
    train_set_x = theano.shared(value=np.asarray(tr_x), borrow=True)
    test_set_x  = theano.shared(value=np.asarray(te_x), borrow=True)
    
    train_set_l = np.asarray(tr_y)
    test_set_l  = np.asarray(te_y)
    
    # compute number of minibatches for training, validation and testing
    n_train_data = train_set_x.get_value(borrow=True).shape[0]
    print "n_train_data: ", n_train_data
    n_test_data = test_set_x.get_value(borrow=True).shape[0]
    print "n_test_data: ", n_test_data

    train_x = np.zeros((n_train_data, N_DIM), dtype=np.float32)
    test_x  = np.zeros((n_test_data, N_DIM), dtype=np.float32)
    
    # loading the trained model
    model_file = file('models/pretrained_model.save', 'rb')
    sda = pkl.load(model_file)
    model_file.close()
    
    get_encoded_data = sda.encoder_function(train_set_x=train_set_x)
    get_single_encoded_data = sda.single_encoder_function()

    for i in range(n_train_data):
        encoded_x = get_encoded_data(index=i)
        if (IS_KMEANS == 1):
            train_x[i] = encoded_x
        # else:
        #     cluster.getDimensionInfo(endoded_x)

    if (IS_KMEANS == 1):
        #flags = cv2.KMEANS_RANDOM_CENTERS
        flags = cv2.KMEANS_PP_CENTERS
        # Apply KMeans
        compactness, labels, centers = cv2.kmeans(data=train_x, K=3, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100000, 0.001), attempts=10, flags=flags)
        #print "Error: ", compactness, len(labels)
        get_accuracy(train_x, train_set_l, labels)

        test_labels = []
        for i in range(n_test_data):
            encoded_x = get_single_encoded_data(train_x=test_set_x.get_value(borrow=True)[i:i+1])
            print "Test Encode: ", encoded_x.shape
            test_labels.append(get_kmeans_labels(centers, encoded_x))
            test_x[i] = encoded_x
        # else:
        #     cluster.getDimensionInfo(endoded_x)
        get_accuracy(test_x, test_set_l, test_labels)

    # else:
    #     time = 1
    #     for data in train_x:
    #         centers[time] = cluster.get_tseries_labels(data,time)
    #         time += 1
    return centers

def get_accuracy(data_x, data_y, labels):
    A = []
    B = []
    C = []
    A_l1 =0
    A_l2 =0
    A_l3 =0
    B_l1 =0
    B_l2 =0
    B_l3 =0
    C_l1 =0
    C_l2 =0
    C_l3 =0
    # Now split the data depending on their labels
    for i in xrange(len(labels)):
        if (labels[i] == 0):
            if data_y[i] == "label1":
                A_l1 += 1
            elif data_y[i] == "label2":
                A_l2 += 1
            elif data_y[i] == "label3":
                A_l3 += 1
            A.append(data_x[i])
        elif (labels[i] == 1):
            if data_y[i] == "label1":
                B_l1 += 1
            elif data_y[i] == "label2":
                B_l2 += 1
            elif data_y[i] == "label3":
                B_l3 += 1
            B.append(data_x[i])
        elif (labels[i] == 2):
            if data_y[i] == "label1":
                C_l1 += 1
            elif data_y[i] == "label2":
                C_l2 += 1
            elif data_y[i] == "label3":
                C_l3 += 1
            C.append(data_x[i])
    print "Length: ", len(A), len(B), len(C)
    len_A = len(A)
    len_B = len(B)
    len_C = len(C)
    
    max_A = max(A_l1,A_l2,A_l3)
    max_B = max(B_l1,B_l2,B_l3)
    max_C = max(C_l1,C_l2,C_l3)

    #accuracy = (max_A/len_A) + (max_B/len_B) + (max_C/len_C)
    accuracy = (max_A + max_B + max_C)/len(labels)
    print "Acc: ", accuracy
    
    print "Cluster A Count: ", A_l1, A_l2, A_l3
    print "Cluster B Count: ", B_l1, B_l2, B_l3
    print "Cluster C Count: ", C_l1, C_l2, C_l3
    

if __name__ == '__main__':
    centers = cluster_train_data()
    label_faces_from_video(centers)
