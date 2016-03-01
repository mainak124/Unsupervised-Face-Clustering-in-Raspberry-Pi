import numpy as np

x1 = np.load('store_faces_neha.npy')
x2 = np.load('store_faces_debanjan.npy')
x3 = np.load('store_faces_akanksha.npy')

x1_train = x1[:900]
x2_train = x2[:900]
x3_train = x3[:900]

x1_test = x1[900:]
x2_test = x2[900:]
x3_test = x3[900:]

train_x = np.concatenate((x1_train,x2_train,x3_train), axis=0)
flat_train_x = np.zeros((train_x.shape[0], train_x.shape[1]*train_x.shape[2]), dtype=np.float32)

test_x = np.concatenate((x1_test,x2_test,x3_test), axis=0)
flat_test_x = np.zeros((test_x.shape[0], test_x.shape[1]*test_x.shape[2]), dtype=np.float32)

for i in xrange(train_x.shape[0]):
	flat_train_x[i] = train_x[i].reshape(1, train_x.shape[1]*train_x.shape[2])

for i in xrange(test_x.shape[0]):
	flat_test_x[i]  = test_x[i].reshape(1, test_x.shape[1]*test_x.shape[2])

flat_train_x = flat_train_x/255
flat_test_x = flat_test_x/255

train_data = flat_train_x - flat_train_x.mean(axis=0)
test_data = flat_test_x - flat_test_x.mean(axis=0)

np.random.shuffle(train_data)
np.random.shuffle(test_data)

np.save("train_faces.npy", train_data)
np.save("test_faces.npy", test_data)
