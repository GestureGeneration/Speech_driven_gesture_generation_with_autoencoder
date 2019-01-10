"""
This file contains an implementation of a particular type of AE,
namely Denoising Autoendoder.

To be used in the files learn_dataset_encoding and train.py

Developed by Taras Kucherenko (tarask@kth.se)
"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from utils.utils import add_noise, loss_reconstruction
from utils.flags import FLAGS


class DAE:
    """ Denoising Autoendoder (DAE)

    More details about the network in the original paper:
    http://www.jmlr.org/papers/v11/vincent10a.html

    The user specifies the structure of this network
    by specifying number of inputs, the number of hidden
    units for each layer and the number of final outputs.
    All this information is set in the utils/flags.py file.

    The number of input neurons is defined as a frame_size*chunk_length,
    since it will take a time-window as an input

    """

    def __init__(self, shape, sess, variance_coef, data_info):
        """DAE initializer

        Args:
          shape:          list of ints specifying
                          num input, hidden1 units,...hidden_n units, num outputs
          sess:           tensorflow session object to use
          varience_coef:  multiplicative factor for the variance of noise wrt the variance of data
          data_info:      key information about the dataset
        """

        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
        self.__variables = {}
        self.__sess = sess

        self.num_hidden_layers = np.size(shape) - 2

        self.batch_size = FLAGS.batch_size
        self.sequence_length = FLAGS.chunk_length

        self.scaling_factor = 1

	    # maximal value and mean pose in the dataset (used for scaling it to interval [-1,1] and back)
        self.max_val = data_info.max_val
        self.mean_pose = data_info.mean_pose


        #################### Add the DATASETS to the GRAPH ###############

        #### 1 - TRAIN ###
        self._train_data_initializer = tf.placeholder(dtype=tf.float32,
                                                      shape=data_info.train_shape)
        self._train_data = tf.Variable(self._train_data_initializer,
                                       trainable=False, collections=[], name='Train_data')
        train_epochs = FLAGS.training_epochs + FLAGS.pretraining_epochs * FLAGS.num_hidden_layers
        train_frames = tf.train.slice_input_producer([self._train_data], num_epochs=train_epochs)
        self._train_batch = tf.train.shuffle_batch(train_frames,
                                                   batch_size=FLAGS.batch_size, capacity=5000,
                                                   min_after_dequeue=1000, name='Train_batch')

        #### 2 - VALIDATE, can be used as TEST ###
        # When optimizing - this dataset stores as a validation dataset,
        # when testing - this dataset stores a test dataset
        self._valid_data_initializer = tf.placeholder(dtype=tf.float32,
                                                      shape=data_info.eval_shape)
        self._valid_data = tf.Variable(self._valid_data_initializer,
                                       trainable=False, collections=[], name='Valid_data')
        valid_frames = tf.train.slice_input_producer([self._valid_data],
                                                     num_epochs=FLAGS.training_epochs)
        self._valid_batch = tf.train.shuffle_batch(valid_frames,
                                                   batch_size=FLAGS.batch_size, capacity=5000,
                                                   min_after_dequeue=1000, name='Valid_batch')

        if FLAGS.weight_decay is not None:
            print('\nWe apply weight decay')

        ### Specify tensorflow setup  ###
        with sess.graph.as_default():

            ##############        SETUP VARIABLES       ######################

            with tf.variable_scope("AE_Variables"):

                for i in range(self.num_hidden_layers + 1):  # go over layers

                    # create variables for matrices and biases for each layer
                    self._create_variables(i, FLAGS.weight_decay)

                ##############        DEFINE THE NETWORK     ##################

                ''' 1 - Setup network for TRAINing '''
                # Input noisy data and reconstruct the original one
                # as in Denoising AutoEncoder
                self._input_ = add_noise(self._train_batch, variance_coef, data_info.data_sigma)
                self._target_ = self._train_batch

                # Define output and loss for the training data
                self._output, _, _ = self.construct_graph(self._input_, FLAGS.dropout)
                self._reconstruction_loss = loss_reconstruction(self._output,
                                                                self._target_, self.max_val)
                tf.add_to_collection('losses', self._reconstruction_loss)  # add weight decay loses
                self._loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

                ''' 2 - Setup network for TESTing '''
                self._valid_input_ = self._valid_batch
                self._valid_target_ = self._valid_batch

                # Define output (no dropout)
                self._valid_output, self._encode, self._decode = \
                    self.construct_graph(self._valid_input_, 1)

                # Define loss
                self._valid_loss = loss_reconstruction(self._valid_output,
                                                       self._valid_target_, self.max_val)
    @property
    def session(self):
        """ Interface for the session"""
        return self.__sess

    @property
    def shape(self):
        """ Interface for the shape"""
        return self.__shape

    # Make more comfortable interface to the network weights

    def _w(self, n, suffix=""):
        return self["matrix"+str(n)+suffix]

    def _b(self, n, suffix=""):
        return self["bias"+str(n)+suffix]

    @staticmethod
    def _feedforward(x, w, b):
        """
        Traditional feedforward layer: multiply on weight matrix, add bias vector
         and apply activation function

        Args:
            x: input ( usually - batch of vectors)
            w: matrix to be multiplied on
            b: bias to be added

        Returns:
            y: result of applying this feedforward layer
        """

        y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w), b))
        return y

    def construct_graph(self, input_seq_pl, dropout):

        """ Construct a TensorFlow graph for the AutoEncoding network

        Args:
          input_seq_pl:     tf placeholder for input data: size [batch_size, sequence_length * DoF]
          dropout:          how much of the input neurons will be activated, value in range [0,1]
        Returns:
          output:           output tensor: result of running input placeholder through the network
          middle_layer:     tensor which is encoding input placeholder into a representation
          decoding:         tensor which is decoding a representation back into the input vector
        """

        network_input = input_seq_pl

        curr_layer = tf.reshape(network_input, [self.batch_size,
                                                FLAGS.chunk_length * FLAGS.frame_size])

        numb_layers = self.num_hidden_layers + 1

        with tf.name_scope("Joint_run"):

            # Pass through the network
            for i in range(numb_layers):

                if i == FLAGS.middle_layer:
                    # Save middle layer
                    with tf.name_scope('middle_layer'):
                        middle_layer = tf.identity(curr_layer)

                with tf.name_scope('hidden'+str(i)):

                    # First - Apply Dropout
                    curr_layer = tf.nn.dropout(curr_layer, dropout)

                    w = self._w(i + 1)
                    b = self._b(i + 1)

                    curr_layer = self._feedforward(curr_layer, w, b)

            output = curr_layer

        # Now create a decoding network

        with tf.name_scope("Decoding"):

            layer = self._representation = tf.placeholder\
                (dtype=tf.float32, shape=middle_layer.get_shape().as_list(), name="Respres.")

            for i in range(FLAGS.middle_layer, numb_layers):

                with tf.name_scope('hidden' + str(i)):

                    # First - Apply Dropout
                    layer = tf.nn.dropout(layer, dropout)

                    w = self._w(i + 1)
                    b = self._b(i + 1)

                    layer = self._feedforward(layer, w, b)

            decoding = layer

        return output, middle_layer, decoding

    def __getitem__(self, item):
        """Get AutoEncoder tf variable

        Returns the specified variable created by this object.
        Names are weights#, biases#, biases#_out, weights#_fixed,
        biases#_fixed.

        Args:
         item: string, variables internal name
        Returns:
         Tensorflow variable
        """
        return self.__variables[item]

    def __setitem__(self, key, value):
        """Store a TensorFlow variable

        NOTE: Don't call this explicitly. It should
        be used only internally when setting up
        variables.

        Args:
          key: string, name of variable
          value: tensorflow variable
        """
        self.__variables[key] = value

    def _create_variables(self, i, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if 'wd' is specified.
        If 'wd' is None, weight decay is not added for this Variable.

        This function was taken from the web

        Args:
          i: number of hidden layer
          wd: add L2Loss weight decay multiplied by this float.
        Returns:
          Nothing
        """

        # Initialize Train weights
        w_shape = (self.__shape[i], self.__shape[i + 1])
        a = tf.multiply(2.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        name_w = "matrix"+str(i + 1)
        self[name_w] = tf.get_variable("Variables/"+name_w,
                                       initializer=tf.random_uniform(w_shape, -1 * a, a))

        # Add weight to the loss function for weight decay
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(self[name_w]), wd, name='wgt_'+str(i)+'_loss')
            tf.add_to_collection('losses', weight_decay)

        # Add the histogram summary
        tf.summary.histogram(name_w, self[name_w])

        # Initialize Train biases
        name_b = "bias"+str(i + 1)
        b_shape = (self.__shape[i + 1],)
        self[name_b] = tf.get_variable("Variables/"+name_b, initializer=tf.zeros(b_shape))

        if i < self.num_hidden_layers:
            # Hidden layer pretrained weights
            # which are used after pretraining before fine-tuning
            self[name_w + "_pretr"] = tf.get_variable(name="Var/" + name_w + "_pretr", initializer=
                                                      tf.random_uniform(w_shape, -1 * a, a),
                                                      trainable=False)
            # Hidden layer pretrained biases
            self[name_b + "_pretr"] = tf.get_variable("Var/"+name_b+"_pretr", trainable=False,
                                                      initializer=tf.zeros(b_shape))

            # Pretraining output training biases
            name_b_out = "bias" + str(i+1) + "_out"
            b_shape = (self.__shape[i],)
            b_init = tf.zeros(b_shape)
            self[name_b_out] = tf.get_variable(name="Var/"+name_b_out, initializer=b_init,
                                               trainable=True)

    def run_less_layers(self, input_pl, n, is_target=False):
        """Return result of a net after n layers or n-1 layer (if is_target is true)
           This function will be used for the layer-wise pretraining of the AE
        Args:
          input_pl:  TensorFlow placeholder of AE inputs
          n:         int specifying pretrain step
          is_target: bool specifying if required tensor
                      should be the target tensor
                     meaning if we should run n layers or n-1 (if is_target)
        Returns:
          Tensor giving pretraining net result or pretraining target
        """
        assert n > 0
        assert n <= self.num_hidden_layers

        last_output = input_pl

        for i in range(n - 1):
            w = self._w(i + 1, "_pretrained")
            b = self._b(i + 1, "_pretrained")

            last_output = self._feedforward(last_output, w, b)

        if is_target:
            return last_output

        last_output = self._feedforward(last_output, self._w(n), self._b(n))

        out = self._feedforward(last_output, self._w(n), self["bias" + str(n) + "_out"])

        return out
