'''
@author: Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@date: 12.10.2017
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@license : MIT License
'''

import tensorflow as tf
import os
import gzip
import numpy as np
from six.moves import configparser
import seq_convertors
import pickle

class Simple_multy_layer_perceptron(object):

    def train_NN(self, config, train_important_information, valid_important_information):

        ##########################
        ### DATASET
        ##########################
        
        train_data_dir = config.get('directories', 'exp_dir')  + '/train_features_dir' 
        valid_data_dir = config.get('directories', 'exp_dir')  + '/valid_features_dir' 
        NN_dir = config.get('directories', 'exp_dir')  + '/NN_train_dir'

        if not os.path.isdir(NN_dir):
            os.mkdir(NN_dir)

        logdir = NN_dir + '/logdir'

        if not os.path.isdir(logdir):
            os.mkdir(logdir)

        #########################
        ### SETTINGS
        ##########################

        # Hyperparameters
        initial_learning_rate = float(config.get('simple_NN', 'initial_learning_rate'))
        decay_steps = int(config.get('simple_NN', 'decay_steps'))
        decay_rate = float(config.get('simple_NN', 'decay_rate'))

        # Architecture
        n_hidden = int(config.get('simple_NN', 'n_hidden'))
        hid_layer_num = int(config.get('simple_NN', 'hidden_layer_num'))
        n_input = train_important_information['input_dim']
        training_epochs = int(config.get('simple_NN', 'training_epochs'))
        batch_size = int(config.get('simple_NN', 'train_batch_size'))
        valid_batch_total = valid_important_information['valid_batch_total']
        n_classes = train_important_information['num_labels']
        training_batch_total = train_important_information['training_batch_total']
        max_input_length = train_important_information['train_utt_max_length']
        max_target_length = train_important_information['train_label_max_length']

        ##########################
        ### GRAPH DEFINITION
        ##########################

        g = tf.Graph()
        with g.as_default():
    
            with tf.name_scope('input'):   
        
                #create the inputs placeholder
                inputs = tf.placeholder(
                tf.float32, shape=[max_input_length, batch_size, n_input], name='features')

                #the length of all the input sequences
                input_seq_length = tf.placeholder(
                        tf.int32, shape=[batch_size],
                                                         name='input_seq_length')

                #split the 3D input tensor in a list of batch_size*input_dim tensors
                split_inputs = tf.unstack(inputs, name='split_inputs_training_op')

                #convert the sequential data to non sequential data
                nonseq_inputs = seq_convertors.seq2nonseq(split_inputs, input_seq_length,
                                                                           name='inputs-processing')

            with tf.name_scope('target'):

                #reference labels
                targets = tf.placeholder(
                             tf.int32, shape=[max_target_length, batch_size, 1],
                             name='targets')

                #the length of all the output sequences
                target_seq_length = tf.placeholder(
                                   tf.int32, shape=[batch_size],
                                   name='output_seq_length')
                
    
            # Model parameters
            with tf.name_scope("weights"):
			
                weights = {'h'+str(i): tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.1), name = "h"+str(i)+"_value") for i in range(2,hid_layer_num + 1)}
                weights['h1'] = tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1), name = "h1_value")
                weights['out'] = tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev=0.1), name = "weight_out_value")

            with tf.name_scope("biases"):

                biases = {'b'+str(i): tf.Variable(tf.zeros([n_hidden]), name = "b"+str(i)+"_value") for i in range(1,hid_layer_num + 1)}
                biases['out'] = tf.Variable(tf.zeros([n_classes]), name = "bias_out_value")

            # Multilayer perceptron

            with tf.name_scope("layer-1"):

                layer_1 = tf.add(tf.matmul(nonseq_inputs, weights['h1']), biases['b1'])

                layer_out = tf.nn.tanh(layer_1)

            for i in range(2,hid_layer_num+1):
                
                with tf.name_scope("layer-"+str(i)):

                    layer = tf.add(tf.matmul(layer_out, weights['h'+str(i)]), biases['b'+str(i)])
                    
                    layer_out = tf.nn.tanh(layer)

                    print "hidden layer " + str(i)


            with tf.name_scope("hid_out"):
    
                nonseq_logits = tf.add(tf.matmul(layer_out, weights['out']), biases['out'])

            with tf.name_scope("targets-processing"):

                #split the 3D targets tensor in a list of batch_size*1 tensors
                split_targets = tf.unstack(targets)

                nonseq_targets = seq_convertors.seq2nonseq(split_targets, target_seq_length,
                                                                               name="targets-processing")
                #make a vector out of the targets
                nonseq_targets = tf.reshape(nonseq_targets, [-1])

                #one hot encode the targets
                #pylint: disable=E1101
                end_nonseq_targets = tf.one_hot(nonseq_targets, int(nonseq_logits.get_shape()[1]))

            with tf.name_scope('soft_max'):

                # Loss and optimizer
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=nonseq_logits, labels=end_nonseq_targets)
                cost = tf.reduce_mean(loss, name='cost_op')

            with tf.name_scope('train'):

                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                           decay_steps, decay_rate, staircase=True)
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train = optimizer.minimize(cost, global_step=global_step, name='train_op')

            with tf.name_scope('Accuracy'):

                # Prediction
                correct_prediction = tf.equal(tf.argmax(end_nonseq_targets, 1), tf.argmax(nonseq_logits, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
                accuracy_valid = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='valid-accuracy_op')

            #create a summary for our cost and accuracy
            tf.summary.scalar("cost", cost)
            tf.summary.scalar("train-accuracy", accuracy)
            tf.summary.scalar("valid-accuracy", accuracy_valid)
            tf.summary.histogram('histogram-train-accuracy', accuracy)
            tf.summary.histogram('histogram-valid-accuracy', accuracy_valid)

            # merge all summaries into a single "operation" which we can execute in a session 
            summary_op = tf.summary.merge_all()

            saver = tf.train.Saver(max_to_keep=10000)


        ##########################
        ### TRAINING & EVALUATION
        ##########################

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95


        with tf.Session(graph=g, config=config) as sess:

            sess.run(tf.global_variables_initializer())

            # create log writer object
            writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
       
            for epoch in range(training_epochs):

                avg_cost = 0.

       
                for i in range(training_batch_total):

                    train_batch_x = np.load(gzip.GzipFile(train_data_dir+'/batch_inputs_'+str(i)+'.npy.gz', "r"))
                    train_batch_y = np.load(gzip.GzipFile(train_data_dir+'/batch_targets_'+str(i)+'.npy.gz', "r"))
                    train_input_seq_length = np.load(gzip.GzipFile(train_data_dir+'/batch_input_seq_length_'+str(i)+'.npy.gz', "r"))
                    train_target_seq_length = np.load(gzip.GzipFile(train_data_dir+'/batch_output_seq_length_'+str(i)+'.npy.gz', "r"))
                    
                    # perform the operations we defined earlier on batch
                    _, c, summary = sess.run([train, cost, summary_op], feed_dict={inputs:  train_batch_x,
                                                                                   targets: train_batch_y,
                                                                                   input_seq_length:  train_input_seq_length,
                                                                                   target_seq_length: train_target_seq_length})
                    avg_cost += c
                    # write log
                    writer.add_summary(summary, epoch * training_batch_total + i)

                train_acc = 0
                for j in range(training_batch_total):

                    train_x = np.load(gzip.GzipFile(train_data_dir+'/batch_inputs_'+str(j)+'.npy.gz', "r"))
                    train_y = np.load(gzip.GzipFile(train_data_dir+'/batch_targets_'+str(j)+'.npy.gz', "r"))
                    train_x_seq_length = np.load(gzip.GzipFile(train_data_dir+'/batch_input_seq_length_'+str(j)+'.npy.gz', "r"))
                    train_y_seq_length = np.load(gzip.GzipFile(train_data_dir+'/batch_output_seq_length_'+str(j)+'.npy.gz', "r"))
                    
                    train_batch_acc = sess.run(accuracy, feed_dict={inputs: train_x,
                                                        targets: train_y,
                                                        input_seq_length: train_x_seq_length,
                                                        target_seq_length: train_y_seq_length})

                    train_acc += train_batch_acc
                    print "batch accuracy " + str(j)

                train_acc /= (training_batch_total)

                valid_acc = 0
                for j in range(valid_batch_total):

                    validation_x = np.load(gzip.GzipFile(valid_data_dir+'/batch_inputs_'+str(j)+'.npy.gz', "r"))
                    validation_y = np.load(gzip.GzipFile(valid_data_dir+'/batch_targets_'+str(j)+'.npy.gz', "r"))
                    validation_x_seq_length = np.load(gzip.GzipFile(valid_data_dir+'/batch_input_seq_length_'+str(j)+'.npy.gz', "r"))
                    validation_y_seq_length = np.load(gzip.GzipFile(valid_data_dir+'/batch_output_seq_length_'+str(j)+'.npy.gz', "r"))
                    
                    validation_batch_acc = sess.run(accuracy_valid, feed_dict={inputs: validation_x,
                                                        targets: validation_y,
                                                        input_seq_length: validation_x_seq_length,
                                                        target_seq_length: validation_y_seq_length})

                    valid_acc += validation_batch_acc

                valid_acc /= valid_batch_total

                #print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
                #print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
				
                accuracy_log_file = open(logdir+'/accuracy_log', "a")
                accuracy_log_file.write("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)))
                accuracy_log_file.write(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc)+'\n')
                accuracy_log_file.close()

                saver.save(sess, NN_dir + '/model.ckpt', global_step=epoch+1)
