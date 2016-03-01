import numpy as np
import cv2

filename = 'store_faces.npy'

train_x = np.load(filename)
mean, eigenvectors = cv.PCACompute(train_x, np.mean(train_x, axis=0).reshape(1,-1))

print 'eigenvectors: ', eigenvectors.shape
