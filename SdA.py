#from __future__ import print_function

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA


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
        """ This class is made to support a variable number of layers.

        :type np_rng: np.random.RandomState
        :param np_rng: np random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
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
            # add the layer to our list of layers
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
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

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

    def build_finetune_functions(self, datasets, batch_size, learning_rate):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score


def test_SdA(finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1):

	train_set_x = theano.shared(value = np.load('new_data/train_faces.npy'), borrow=True)
	test_set_x  = theano.shared(value = np.load('new_data/test_faces.npy'), borrow=True)
	
	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_train_batches //= batch_size
	
	# np random generator
	# start-snippet-3
	np_rng = np.random.RandomState(89677)
	print('... building the model')
	# construct the stacked denoising autoencoder class
	sda = SdA(
	    np_rng=np_rng,
	    n_ins=30 * 30,
	    hidden_layers_sizes=[500, 250, 100],
	    n_outs=10
	)
	# end-snippet-3 start-snippet-4
	#########################
	# PRETRAINING THE MODEL #
	#########################
	print('... getting the pretraining functions')
	pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
	                                            batch_size=batch_size)
	
	print('... pre-training the model')
	start_time = timeit.default_timer()
	## Pre-train layer-wise
	corruption_levels = [.1, .2, .3]
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
	
	# print(('The pretraining code for file ' +
	#        os.path.split(__file__)[1] +
	#        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
	print(('The pretraining code for file ' +
	       ' ran for %.2fm' % ((end_time - start_time) / 60.)))

	# end-snippet-4
	########################
	# FINETUNING THE MODEL #
	########################
	
	# # get the training, validation and testing function for the model
	# print('... getting the finetuning functions')
	# train_fn, validate_model, test_model = sda.build_finetune_functions(
	#     datasets=datasets,
	#     batch_size=batch_size,
	#     learning_rate=finetune_lr
	# )
	
	# print('... finetunning the model')
	# # early-stopping parameters
	# patience = 10 * n_train_batches  # look as this many examples regardless
	# patience_increase = 2.  # wait this much longer when a new best is
	#                         # found
	# improvement_threshold = 0.995  # a relative improvement of this much is
	#                                # considered significant
	# validation_frequency = min(n_train_batches, patience // 2)
	#                               # go through this many
	#                               # minibatche before checking the network
	#                               # on the validation set; in this case we
	#                               # check every epoch
	
	# best_validation_loss = np.inf
	# test_score = 0.
	# start_time = timeit.default_timer()
	
	# done_looping = False
	# epoch = 0
	
	# while (epoch < training_epochs) and (not done_looping):
	#     epoch = epoch + 1
	#     for minibatch_index in range(n_train_batches):
	#         minibatch_avg_cost = train_fn(minibatch_index)
	#         iter = (epoch - 1) * n_train_batches + minibatch_index
	
	#         if (iter + 1) % validation_frequency == 0:
	#             validation_losses = validate_model()
	#             this_validation_loss = np.mean(validation_losses)
	#             print('epoch %i, minibatch %i/%i, validation error %f %%' %
	#                   (epoch, minibatch_index + 1, n_train_batches,
	#                    this_validation_loss * 100.))
	
	#             # if we got the best validation score until now
	#             if this_validation_loss < best_validation_loss:
	
	#                 #improve patience if loss improvement is good enough
	#                 if (
	#                     this_validation_loss < best_validation_loss *
	#                     improvement_threshold
	#                 ):
	#                     patience = max(patience, iter * patience_increase)
	
	#                 # save best validation score and iteration number
	#                 best_validation_loss = this_validation_loss
	#                 best_iter = iter
	
	#                 # test it on the test set
	#                 test_losses = test_model()
	#                 test_score = np.mean(test_losses)
	#                 print(('     epoch %i, minibatch %i/%i, test error of '
	#                        'best model %f %%') %
	#                       (epoch, minibatch_index + 1, n_train_batches,
	#                        test_score * 100.))
	
	#         if patience <= iter:
	#             done_looping = True
	#             break
	
	# end_time = timeit.default_timer()
	# print(
	#     (
	#         'Optimization complete with best validation score of %f %%, '
	#         'on iteration %i, '
	#         'with test performance %f %%'
	#     )
	#     % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
	# )
	# print(('The training code for file ' +
	#        os.path.split(__file__)[1] +
	#        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    test_SdA()
