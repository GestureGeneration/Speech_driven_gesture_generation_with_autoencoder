"""
This module contrains all the flags for the motion representation learning repository
"""
from __future__ import division
import os
from os.path import join as pjoin

import tensorflow as tf

# Modify this function to set your home directory for this repo
def home_out(path):
    return pjoin(os.environ['HOME'], 'tmp', 'MoCap', path)

flags = tf.app.flags
FLAGS = flags.FLAGS

"""  							Fine-tuning Parameters 				"""

#                       Flags about the sequence processing

flags.DEFINE_integer('chunk_length', 1, 'Length of the chunks, for the data processing.')

#                               Flags about training
flags.DEFINE_float('learning_rate', 0.0001,
                   'learning rate for training .')
flags.DEFINE_float('pretraining_learning_rate', 0.001 ,
                   'learning rate for training .')

flags.DEFINE_float('variance_of_noise', 0.05, 'Coefficient for the gaussian noise '
                                              'added to every point in input during the training')

flags.DEFINE_boolean('pretrain', False,' Whether we pretrain the model in a layerwise way')
flags.DEFINE_boolean('restore', False,' Whether we restore the model from the checkpoint')

flags.DEFINE_boolean('evaluate', False, ' Whether we are evaluating the system')

flags.DEFINE_float('dropout', 0.9, 'Probability to keep the neuron on')

flags.DEFINE_integer('batch_size', 128,
                     'Size of the mini batch')

flags.DEFINE_integer('training_epochs', 50,
                     "Number of training epochs for pretraining layers")
flags.DEFINE_integer('pretraining_epochs', 5,
                     "Number of training epochs for pretraining layers")

flags.DEFINE_float('weight_decay', 0.5, ' Whether we apply weight decay')

flags.DEFINE_boolean('early_stopping', True, ' Whether we do early stopping')
flags.DEFINE_float('delta_for_early_stopping', 0.5, 'How much worst the results must get in order'
                                                    ' for training to be terminated.'
                                                    ' 0.05 mean 5% worst than best we had.')

#                       Network Architecture Specific Flags
flags.DEFINE_integer('frame_size', 24, 'Dimensionality of the input for a single frame')

flags.DEFINE_integer("num_hidden_layers", 1, "Number of hidden layers")
flags.DEFINE_integer("middle_layer", 1, "Number of hidden layers")

flags.DEFINE_integer('layer1_width', 20, 'Number of units in each hidden layer ')
flags.DEFINE_integer('layer2_width', 248, 'Number of units in each hidden layer ')
flags.DEFINE_integer('layer3_width', 312, 'Number of units in each hidden layer ')

#                           Constants

flags.DEFINE_integer('seed', 123456, 'Random seed')

flags.DEFINE_string('summary_dir', home_out('summaries_exp'),
                    'Directory to put the summary data')

flags.DEFINE_string('chkpt_dir', home_out('chkpts_exp'),
                    'Directory to put the model checkpoints')

flags.DEFINE_string('results_file', home_out('results.txt'),
                    'File to put the experimental results')
