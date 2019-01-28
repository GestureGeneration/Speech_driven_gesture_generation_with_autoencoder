"""
This file contains an implementation of the training script,
for the AE type of neural network, used for the representation learning.

It is used in files learn_dataset_encoding.py and encode_dataset.py

Developed by Taras Kucherenko (tarask@kth.se)
"""

import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

from DAE import DAE
import utils.utils as ut
import utils.flags as fl

class DataInfo(object):
    """Information about the datasets

     Will be passed to the network for creating corresponding variables in the graph
    """

    def __init__(self, data_sigma, train_shape, eval_shape, max_val, mean_pose):
        """DataInfo initializer

        Args:
          data_sigma:   variance in the dataset
          train_shape:  dimensionality of the train dataset
          eval_shape:   dimensionality of the evaluation dataset
        """
        self.data_sigma = data_sigma
        self.train_shape = train_shape
        self.eval_shape = eval_shape
        self.max_val = max_val
        self.mean_pose = mean_pose


###############################################
####                                    #######
####              TRAIN                 #######
####                                    #######
###############################################

def learning(data, data_info, just_restore=False):
    """ Training of the network

    Args:
        data:           dataset to train on
        data_info :     meta information about this dataset (such as variance, mean pose, etc.)
                        it is an object from the class DataInfo (defined at the top of this file)
        just_restore:   weather we are going to only restore the model from the checkpoint
                        or are we going to train it as well

    Returns:
        nn:             Neural Network trained on a data provided
    """

    test = False
    debug = False

    with tf.Graph().as_default():

        tf.set_random_seed(fl.FLAGS.seed)

        start_time = time.time()

        # Read the flags
        variance = fl.FLAGS.variance_of_noise
        num_hidden = fl.FLAGS.num_hidden_layers
        dropout = fl.FLAGS.dropout
        learning_rate = fl.FLAGS.learning_rate
        batch_size = fl.FLAGS.batch_size

        hidden_shapes = [fl.FLAGS.layer1_width
                         for j in range(num_hidden)]

        # Check if the flags makes sence
        if dropout < 0 or variance < 0:
            print('ERROR! Have got negative values in the flags!')
            exit(1)

        # Allow TensorFlow to change device allocation when needed
        config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
        # Adjust configuration so that multiple executions are possible
        config.gpu_options.allow_growth = True

        # Start a session
        sess = tf.Session(config=config)

        if debug:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, "taras-All-Series:6064")

        # Create a neural network
        shape = [fl.FLAGS.frame_size * fl.FLAGS.chunk_length] + hidden_shapes + [
            fl.FLAGS.frame_size * fl.FLAGS.chunk_length]
        nn = DAE(shape, sess, variance, data_info)
        print('\nDAE with the following shape was created : ', shape)

        # Initialize input_producer
        sess.run(tf.local_variables_initializer())

        max_val = nn.max_val

        with tf.variable_scope("Train"):

            ##############        DEFINE  Optimizer and training OPERATOR      ############

            # Define the optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # Do gradient clipping
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(nn._loss, tvars), 1e12)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                 global_step=tf.train.get_or_create_global_step())

            # Prepare for making a summary for TensorBoard
            train_error = tf.placeholder(dtype=tf.float32, shape=(), name='train_error')
            eval_error = tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')

            train_summary_op = tf.summary.scalar('Train_error', train_error)
            eval_summary_op = tf.summary.scalar('Validation_error', eval_error)

            summary_dir = fl.FLAGS.summary_dir
            summary_writer = tf.summary.FileWriter(summary_dir, graph=tf.get_default_graph())

            num_batches = int(data.train.num_sequences / batch_size)

            # Initialize the part of the graph with the input data
            sess.run(nn._train_data.initializer,
                     feed_dict={nn._train_data_initializer: data.train.sequences})
            sess.run(nn._valid_data.initializer,
                     feed_dict={nn._valid_data_initializer: data.test.sequences})

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if fl.FLAGS.pretrain:
                layers_amount = len(nn.shape) - 2

                # create an optimizers
                pretrain_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

                # Make an array of the trainers for all the layers
                trainers = [pretrain_optimizer.minimize(
                    ut.loss_reconstruction(nn.run_less_layers(nn._input_, i+1),
                                           nn.run_less_layers(nn._input_, i+1, is_target=True),
                                           max_val, pretrain=True),
                    global_step=tf.train.get_or_create_global_step(),
                    name='Layer_wise_optimizer_'+str(i))
                            for i in range(layers_amount)]

                # Initialize all the variables
                sess.run(tf.global_variables_initializer())

            else:
                print("Initializing variables ...\n")
                sess.run(tf.global_variables_initializer())

            # Create a saver
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            chkpt_file = fl.FLAGS.chkpt_dir + '/chkpt-final'

            # restore model, if needed
            if fl.FLAGS.restore:
                saver.restore(sess, chkpt_file)
                print("Model restored from the file " + str(chkpt_file) + '.')

            if just_restore:
                coord.request_stop()
                return nn

            # A few initialization for the early stopping
            delta = fl.FLAGS.delta_for_early_stopping  # error tolerance for early stopping
            best_error = 10000
            num_valid_batches = int(data.test.num_sequences / batch_size)

            try:  # running enqueue threads.

                # Pretrain
                if fl.FLAGS.pretrain:
                    layerwise_pretrain(nn, trainers, layers_amount, num_batches)

                # Train the whole network jointly
                step = 0
                print('\nFinetune the whole network on ', num_batches, ' batches with ', batch_size,
                      ' training examples in each for', fl.FLAGS.training_epochs, ' epochs...')
                print("")
                print(" ______________ ______")
                print("|     Epoch    | RMSE |")
                print("|------------  |------|")

                while not coord.should_stop():
                    _, train_error_ = sess.run([train_op, nn._reconstruction_loss], feed_dict={})

                    if step % num_batches == 0:
                        epoch = step * 1.0 / num_batches

                        train_summary = sess.run(train_summary_op, feed_dict={
                            train_error: np.sqrt(train_error_)})

                        # Print results of screen
                        epoch_str = "| {0:3.0f} ".format(epoch)[:5]
                        perc_str = "({0:3.2f}".format(epoch*100.0 / fl.FLAGS.training_epochs)[:5]
                        error_str = "%) |{0:5.2f}".format(train_error_)[:10] + "|"
                        print(epoch_str, perc_str, error_str)

                        if epoch % 5 == 0 and test:

                            rmse = test(nn, fl.FLAGS.data_dir + '/test_1.binary')
                            print("\nOur RMSE for the first test sequence is : ", rmse)

                            rmse = test(nn, fl.FLAGS.data_dir + '/test_2.binary')
                            print("\nOur RMSE for the second test sequenceis : ", rmse)

                        if epoch > 0:
                            summary_writer.add_summary(train_summary, step)

                            # Evaluate on the validation sequences
                            error_sum = 0
                            for valid_batch in range(num_valid_batches):
                                curr_err = sess.run([nn._valid_loss], feed_dict={})
                                error_sum += curr_err[0]
                            new_error = error_sum / (num_valid_batches)
                            eval_sum = sess.run(eval_summary_op,
                                                feed_dict={eval_error: np.sqrt(new_error)})
                            summary_writer.add_summary(eval_sum, step)

                            # Early stopping
                            if fl.FLAGS.early_stopping:
                                if (new_error - best_error) / best_error > delta:
                                    print('After ' + str(step) + ' steps started overfitting')
                                    break
                                if new_error < best_error:
                                    best_error = new_error

                                    # Saver for the model
                                    save_path = saver.save(sess, chkpt_file)

                            if epoch % 5 == 0:
                                # Save for the model
                                save_path = saver.save(sess, chkpt_file)
                                print('Done training for %d epochs' % (epoch))
                                print("The model was saved in file: %s" % save_path)

                    step += 1

            except tf.errors.OutOfRangeError:
                if not fl.FLAGS.early_stopping:
                    # Save the model
                    save_path = saver.save(sess, chkpt_file)
                print('Done training for %d epochs, %d steps.' % (fl.FLAGS.training_epochs, step))
                print("The final model was saved in file: %s" % save_path)
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)

        duration = (time.time() - start_time) / 60  # in minutes, instead of seconds

        print("The training was running for %.3f  min" % (duration))

        return nn

###############################################
####                                    #######
####         INTERFACE FUNCTIONS        #######
####                                    #######
###############################################

def encode(nn, input_seq):
    """ Obtaining a representation from AE (AutoEncoder)

      Args:
          nn:          trained AutoEncoder
          input_seq:   input sequence to be encoded

      Returns:
          output_vec:  vector of encoding
    """

    print("Encoding ...")

    with nn.session.graph.as_default() as sess:

        # Obtain important constants

        sess = nn.session
        mean_pose = nn.mean_pose
        max_val = nn.max_val

        #                    GET THE DATA

        # get input sequnce

        Preprocess = False
        if Preprocess:
            coords_minus_mean = input_seq - mean_pose[np.newaxis, :]
            eps = 1e-15
            coords_normalized = np.divide(coords_minus_mean, max_val[np.newaxis, :] + eps)
        else:
            coords_normalized = input_seq

        # Check if we can cut sequence into the chunks of length ae.sequence_length
        if coords_normalized.shape[0] < nn.sequence_length:
            mupliplication_factor = int(nn.batch_size * nn.sequence_length /
                                        coords_normalized.shape[0]) + 1

            # Pad the sequence with itself in order to fill the sequence completely
            coords_normalized = np.tile(coords_normalized, (mupliplication_factor, 1))
            print("Test sequence was way too short!")

        # Split it into chunks

        all_chunks = np.reshape([coords_normalized],
                                (-1, fl.FLAGS.frame_size*fl.FLAGS.chunk_length))

        if all_chunks.shape[0] < nn.batch_size:
            mupliplication_factor = int(nn.batch_size / all_chunks.shape[0]) + 1

            # Pad the sequence with itself in order to fill the batch completely
            all_chunks = np.tile(all_chunks, (mupliplication_factor, 1))

        # Batch those chunks
        batches = np.array([all_chunks[i:i + nn.batch_size, :]
                            for i in range(0, len(all_chunks) - nn.batch_size + 1, nn.batch_size)])

        numb_of_batches = batches.shape[0]

        #                    RUN THE NETWORK

        output_batches = np.array([])

        # Go over all batches one by one
        for batch_numb in range(numb_of_batches):
            output_batch = sess.run([nn._encode],
                                    feed_dict={nn._valid_input_: batches[batch_numb]})

            output_batches = np.append(output_batches, output_batch, axis=0) \
                if output_batches.size else np.array(output_batch)

        # Postprocess...
        output_vec = np.reshape(output_batches, (-1, output_batches.shape[-1]))

        return output_vec

def decode(nn, represent_vec):
    """ Decoding a representation from AE (AutoEncoder)

      Args:
          nn:              trained AutoEncoder
          represent_vec:   input sequence to be encoded

      Returns:
          output_seq:  vector of encoding
    """

    print("Decoding ...")

    with nn.session.graph.as_default() as sess:

        # Obtain important constants

        sess = nn.session
        mean_pose = nn.mean_pose
        max_val = nn.max_val

        #                    GET THE DATA

        # Check if we can cut sequence into the chunks of length ae.sequence_length
        if represent_vec.shape[0] < nn.sequence_length:
            mupliplication_factor = int(nn.batch_size * nn.sequence_length /
                                        represent_vec.shape[0]) + 1

            # Pad the sequence with itself in order to fill the sequence completely
            represent_vec = np.tile(represent_vec, (mupliplication_factor, 1))
            print("Test sequence was way too short, so we padded it with itself!")

        # Split it into chunks
        all_chunks = represent_vec

        if all_chunks.shape[0] < nn.batch_size:
            mupliplication_factor = int(nn.batch_size / all_chunks.shape[0]) + 1

            # Pad the sequence with itself in order to fill the batch completely
            all_chunks = np.tile(all_chunks, (mupliplication_factor, 1))

        # Batch those chunks
        batches = np.array([all_chunks[i:i + nn.batch_size, :]
                            for i in range(0, len(all_chunks) - nn.batch_size + 1, nn.batch_size)])

        numb_of_batches = batches.shape[0]

        #                    RUN THE NETWORK

        output_batches = np.array([])

        # Go over all batches one by one
        for batch_numb in range(numb_of_batches):
            output_batch = sess.run([nn._decode],
                                    feed_dict={nn._representation: batches[batch_numb]})
            output_batches = np.append(output_batches, output_batch, axis=0) \
                if output_batches.size else np.array(output_batch)

        # Postprocess...
        output_vec = np.reshape(output_batches, (-1, fl.FLAGS.chunk_length * fl.FLAGS.frame_size))

        # Convert back to original values
        reconstructed = ut.convert_back_to_3d_coords(output_vec, max_val, mean_pose)

        return reconstructed

###############################################
####                                    #######
####      LAYERWISE PRETRAINING         #######
####                                    #######
###############################################

def layerwise_pretrain(nn, trainers, layers_amount, num_batches):
    """
    Pretrain AutoEncoding neural network in a layer-wise way
    Args:
        nn:            neural network to be trained
        trainers:      optimizers to be used
        layers_amount: amount of layers in the network
        num_batches:   number of batches

    Returns:
        nn:            pretrained trained neural network

    """
    sess = nn.session

    debug = False

    for i in range(layers_amount):
        n = i + 1
        print('Pretraining layer number ', n, ' for ', fl.FLAGS.pretraining_epochs, ' epochs ... ')

        with tf.variable_scope("layer_{0}".format(n)):

            layer = nn.run_less_layers(nn._input_, n)

            with tf.name_scope("pretraining_loss"):
                target_for_loss = nn.run_less_layers(nn._input_, n, is_target=True)

            loss = ut.loss_reconstruction(target_for_loss, layer, None, pretrain=True),

            pretrain_trainer = trainers[i]

            for steps in range(num_batches * fl.FLAGS.pretraining_epochs):

                loss_summary, loss_value = sess.run([pretrain_trainer, loss])

                if debug:
                    if steps%num_batches == 0:
                        print("After "+ str(steps/num_batches)+" epochs loss is "+ str(loss_value))

            # Copy the trained weights to the fixed matrices and biases
            nn['matrix'+str(n)+ '_pretrained'] = nn._w(n)
            nn['bias'+str(n)+ '_pretrained'] = nn._b(n)
