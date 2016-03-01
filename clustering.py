import numpy as np
import cPickle as pkl
from stacked_autoencoder import SdA
import theano


def cluster_data():

    train_set_x = theano.shared(value = np.load('new_data/train_faces.npy'), borrow=True)
    test_set_x  = theano.shared(value = np.load('new_data/test_faces.npy'), borrow=True)
    
    # compute number of minibatches for training, validation and testing
    n_data = train_set_x.get_value(borrow=True).shape[0]
    
    # loading the trained model
    model_file = file('models/pretrained_model.save', 'rb')
    sda = pkl.load(model_file)
    model_file.close()
    
    get_encoded_data = sda.encoder_function(train_set_x=train_set_x)

    for i in range(n_data):
        encoded_x = get_encoded_data(index=i)
        print "Shape: ", encoded_x.shape
        print "Type: ", encoded_x.dtype


if __name__ == '__main__':
    cluster_data()
