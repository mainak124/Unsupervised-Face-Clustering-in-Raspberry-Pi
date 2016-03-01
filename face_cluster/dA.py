from __future__ import print_function

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

# from logistic_sgd import load_data
# from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


class dA(object):

    def __init__(
        self,
        np_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

	# def rmsprop(lr, tparams, grads, x, mask, y, cost):
	#     """
	#     A variant of  SGD that scales the step size by running average of the
	#     recent step norms.
	# 
	#     Parameters
	#     ----------
	#     lr : Theano SharedVariable
	#         Initial learning rate
	#     tpramas: Theano SharedVariable
	#         Model parameters
	#     grads: Theano variable
	#         Gradients of cost w.r.t to parameres
	#     x: Theano variable
	#         Model inputs
	#     mask: Theano variable
	#         Sequence mask
	#     y: Theano variable
	#         Targets
	#     cost: Theano variable
	#         Objective fucntion to minimize
	# 
	#     Notes
	#     -----
	#     For more information, see [Hint2014]_.
	# 
	#     .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
	#        lecture 6a,
	#        http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
	#     """
	# 
	#     zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),
	#                                   name='%s_grad' % k)
	#                     for k, p in tparams.items()]
	#     running_grads = [theano.shared(p.get_value() * np_floatX(0.),
	#                                    name='%s_rgrad' % k)
	#                      for k, p in tparams.items()]
	#     running_grads2 = [theano.shared(p.get_value() * np_floatX(0.),
	#                                     name='%s_rgrad2' % k)
	#                       for k, p in tparams.items()]
	# 
	#     zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	#     rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
	#     rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
	#              for rg2, g in zip(running_grads2, grads)]
	# 
	#     f_grad_shared = theano.function([x, mask, y], cost,
	#                                     updates=zgup + rgup + rg2up,
	#                                     name='rmsprop_f_grad_shared')
	# 
	#     updir = [theano.shared(p.get_value() * np_floatX(0.),
	#                            name='%s_updir' % k)
	#              for k, p in tparams.items()]
	#     updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
	#                  for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
	#                                             running_grads2)]
	#     param_up = [(p, p + udn[1])
	#                 for p, udn in zip(tparams.values(), updir_new)]
	#     f_update = theano.function([lr], [], updates=updir_new + param_up,
	#                                on_unused_input='ignore',
	#                                name='rmsprop_f_update')
	# 
	#     return f_grad_shared, f_update


    def get_cost_updates(self, corruption_level, learning_rate):

		#tilde_x = self.get_corrupted_input(self.x, corruption_level)
		tilde_x = self.x
		y = self.get_hidden_values(tilde_x)
		z = self.get_reconstructed_input(y)
		L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
		cost = T.mean(L)
		
		gparams = T.grad(cost, self.params)
		updates = [
		    (param, param - learning_rate * gparam)
		    for param, gparam in zip(self.params, gparams)
		]
		
		return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=100,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

	# datasets = load_data(dataset)
	# train_set_x, train_set_y = datasets[0]
	
	train_set_x = theano.shared(value = np.load('train_faces1.npy'), borrow=True)
	test_set_x  = theano.shared(value = np.load('test_faces1.npy'), borrow=True)
	
	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
	
	# allocate symbolic variables for the data
	index = T.lscalar()    # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	
	if not os.path.isdir(output_folder):
	    os.makedirs(output_folder)
	os.chdir(output_folder)
	
	####################################
	# BUILDING THE MODEL NO CORRUPTION #
	####################################
	
	rng = np.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))
	
	da = dA(
	    np_rng=rng,
	    theano_rng=theano_rng,
	    input=x,
	    n_visible=30 * 30,
	    n_hidden=500
	)
	
	cost, updates = da.get_cost_updates(
	    corruption_level=0.,
	    learning_rate=learning_rate
	)
	
	train_da = theano.function(
	    [index],
	    cost,
	    updates=updates,
	    givens={
	        x: train_set_x[index * batch_size: (index + 1) * batch_size]
	    }
	)
	
	start_time = timeit.default_timer()
	
	############
	# TRAINING #
	############
	
	# go through training epochs
	for epoch in range(training_epochs):
	    # go through trainng set
	    c = []
	    for batch_index in range(n_train_batches):
	        c.append(train_da(batch_index))
	
	    print('Training epoch %d, cost ' % epoch, np.mean(c))
	
	end_time = timeit.default_timer()
	
	training_time = (end_time - start_time)

if __name__ == '__main__':
    test_dA()
