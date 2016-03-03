#from __future__ import print_function

import os
import sys
import timeit

import numpy as np
import cPickle as pkl

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from autoencoder import dA


# start-snippet-1
class SdA(object):

    def __init__(
        self,
        np_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x') 

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=np_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to the list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            dA_layer = dA(np_rng=np_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        self.out = self.sigmoid_layers[-1].output

        # self.params.extend(self.logLayer.params)
        # z = 
		# L = T.sum(((self.x - z) ** 2) , axis=1)
        # self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size
        
        pretrain_fns = []
        for dA in self.dA_layers:
        	# get the cost and the updates list
        	cost, updates = dA.get_cost_updates(corruption_level,
        	                                    learning_rate)
        	# compile the theano function
        	fn = theano.function(
        		inputs=[
        			index,
        			corruption_level,
        			learning_rate
        			# theano.In(corruption_level, value=0.2),
        			# theano.In(learning_rate, value=0.1)
        		],
        		outputs=cost,
        		updates=updates,
        		givens={
        			self.x: train_set_x[batch_begin: batch_end]
        		}
        	)
        	# append `fn` to the list of functions
        	pretrain_fns.append(fn)
        
        return pretrain_fns
        
    def encoder_function(self, train_set_x):

        index = T.lscalar('index')
        # compile the theano function
        get_encoded_data = theano.function(
            [index],
            outputs=self.out,
            givens={
                self.x: train_set_x[index: index+1]
            }
        )
        return get_encoded_data

    def single_encoder_function(self):

        train_x = T.matrix('train_x')
        # compile the theano function
        get_single_encoded_data = theano.function(
            [train_x],
            outputs=self.out,
            givens={
                self.x: train_x
            },
            allow_input_downcast=True
        )
        return get_single_encoded_data


def test_SdA(finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             batch_size=20):

    train_set = np.load('new_data/train_faces.npy')
    test_set = np.load('new_data/test_faces.npy')

    tr_x = [i[0] for i in train_set]
    te_x = [i[0] for i in test_set]
    
    train_set_x = theano.shared(value=np.asarray(tr_x), borrow=True)
    test_set_x  = theano.shared(value=np.asarray(te_x), borrow=True)
    # train_set_x = theano.shared(value = np.load('new_data/train_faces.npy'), borrow=True)
    # test_set_x  = theano.shared(value = np.load('new_data/test_faces.npy'), borrow=True)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    
    np_rng = np.random.RandomState(89677)
    print('... building the model')
    # construct the stacked denoising autoencoder class
    sda = SdA(
        np_rng=np_rng,
        n_ins=30 * 30,
        hidden_layers_sizes=[500, 250, 100],
        n_outs=10
    )
    
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    
    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [0.0, 0.0, 0.0]
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                        corruption=corruption_levels[i],
                        lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c)))
    
    end_time = timeit.default_timer()
    
    print(('The pretraining code for file ' +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    
    f = file('models/pretrained_model.save', 'wb')
    pkl.dump(sda, f, protocol=pkl.HIGHEST_PROTOCOL)
    f.close()

if __name__ == '__main__':
    test_SdA()
