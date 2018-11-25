'''
@author: Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@date: 21.12.2017
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
from CnnLayer import CnnLayer

class Vanila_conv_net(object):

    def convolution(self, inputs_img, name_layer, in_dim, out_dim, t_conv_size, f_conv_size, training_phase):
        with tf.name_scope('parameters_'+name_layer):
            n = t_conv_size*f_conv_size*out_dim
            weights = tf.get_variable('weights_'+name_layer, [t_conv_size, f_conv_size, in_dim, out_dim],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            biases = tf.get_variable('biases_'+name_layer,   [out_dim],   initializer=tf.constant_initializer(0) )

        with tf.name_scope('conv_'+name_layer):
            conv = tf.nn.conv2d(inputs_img,  weights, [1, 1, 1, 1], padding='VALID')
            #print conv.get_shape()
            #conv = tf.layers.batch_normalization(conv, training=True, reuse=tf.AUTO_REUSE,  name='batch_norm')
            conv = tf.layers.batch_normalization(conv, training=training_phase)
            hidden = tf.nn.relu(conv + biases)

            print 'hidden_'+ name_layer
            print hidden.get_shape()

        return hidden


    def train_NN(self, config, train_important_information, valid_important_information):

        ##########################
        ### DATASET
        ##########################
        
        train_data_dir = config.get('directories', 'exp_dir')  + '/train_features_dir'
        valid_data_dir = config.get('directories', 'exp_dir')  + '/valid_features_dir'
        NN_dir = config.get('directories', 'exp_dir')  + '/CNN_train_dir'

        if not os.path.isdir(NN_dir):
            os.mkdir(NN_dir)

        logdir = NN_dir + '/logdir'

        if not os.path.isdir(logdir):
            os.mkdir(logdir)

        #########################
        ### SETTINGS
        ##########################

        # Hyperparameters
        learning_rate = float(config.get('simple_NN', 'initial_learning_rate'))
        #decay_steps = int(config.get('simple_NN', 'decay_steps')) # we are using num_steps instead of this
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

        num_steps = training_epochs * training_batch_total
        valid_frequency = training_batch_total #means after each epoch
        total_number_of_retries = 3

        #batch_size = 14 #ommit

        ##########################
        ### GRAPH DEFINITION
        ##########################

        g = tf.Graph()
        with g.as_default():

            # Batchnorm settings. As we are calculating training accuracy and training at the same time so we are keeping the following comment on.
            training_phase = tf.placeholder(tf.bool, None, name='training_phase')

            #for making the learning rate half
            initial_learning_rate = tf.placeholder(tf.float32, None, name='initial_l_rate')
            learning_rate_factor = tf.placeholder(tf.float32, None, name='factor_value')
    
            with tf.name_scope('input'):   
        
                #create the inputs placeholder
                inputs = tf.placeholder(
                tf.float32, shape=[max_input_length, batch_size, n_input], name='features')

                #the length of all the input sequences
                input_seq_length = tf.placeholder(tf.int32, shape=[batch_size], name='input_seq_length')

                #split the 3D input tensor in a list of batch_size*input_dim tensors
                split_inputs = tf.unstack(inputs, name='split_inputs_training_op')

                #convert the sequential data to non sequential data
                nonseq_inputs = seq_convertors.seq2nonseq(split_inputs, input_seq_length, name='inputs-processing')
            
            print 'Layer: '
            print 'Input: '
            print nonseq_inputs.get_shape()

            with tf.name_scope('prep_data_l1'):
                inputs_img = tf.reshape(nonseq_inputs, tf.stack( [ tf.shape(nonseq_inputs)[0] , 7, 1, 13] )  )
                inputs_img = tf.transpose(inputs_img, [ 0 , 1, 3, 2 ] )

            print 'Input Img: '
            print inputs_img.get_shape().as_list()

            hidden_1 = self.convolution(inputs_img, 'conv_l1', 1, 256, 4, 4, training_phase)

            with tf.name_scope('pool_l1'):
                pool = tf.nn.max_pool(hidden_1, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

            print 'poll_l1: '
            print pool.get_shape().as_list()

            hidden_2 = self.convolution(pool, 'conv_l2', 256, 256, 3, 4, training_phase)

            with tf.name_scope('out_op'):
                shape = hidden_2.get_shape().as_list()
                outputs = tf.reshape(hidden_2, tf.stack( [tf.shape(hidden_2)[0], shape[1]  * shape[2]  * shape[3]   ] ) )

            print 'Outputs: '
            print outputs.get_shape().as_list()          
            
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
                weights['h1'] = tf.Variable(tf.truncated_normal([outputs.get_shape().as_list()[1], n_hidden], stddev=0.1), name = "h1_value")
                weights['out'] = tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev=0.1), name = "weight_out_value")

            with tf.name_scope("biases"):

                biases = {'b'+str(i): tf.Variable(tf.zeros([n_hidden]), name = "b"+str(i)+"_value") for i in range(1,hid_layer_num + 1)}
                biases['out'] = tf.Variable(tf.zeros([n_classes]), name = "bias_out_value")

            # Multilayer perceptron

            with tf.name_scope("layer-1"):
			
                layer_1 = tf.add(tf.matmul(outputs, weights['h1']), biases['b1'])

                layer_out = tf.nn.tanh(layer_1)


            for i in range(2,hid_layer_num+1):
                
                with tf.name_scope("layer-"+str(i)):

                    layer = tf.add(tf.matmul(layer_out, weights['h'+str(i)]), biases['b'+str(i)])
                    
                    layer_out = tf.nn.tanh(layer)

            
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
                validation_loss = tf.nn.softmax_cross_entropy_with_logits(logits=nonseq_logits, labels=end_nonseq_targets)
                validation_cost = tf.reduce_mean(validation_loss, name='validation_cost_op')

                train_loss = tf.nn.softmax_cross_entropy_with_logits(logits=nonseq_logits, labels=end_nonseq_targets)
                train_cost = tf.reduce_mean(train_loss, name='train_cost_op')

            with tf.name_scope('train'):

                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, num_steps, decay_rate, staircase=True) * learning_rate_factor
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train = optimizer.minimize(train_cost, global_step=global_step, name='train_op')

            with tf.name_scope('Accuracy'):

                # Prediction
                correct_prediction = tf.equal(tf.argmax(end_nonseq_targets, 1), tf.argmax(nonseq_logits, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
                accuracy_valid = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='valid-accuracy_op')

            #create a summary for our cost and accuracy
            tf.summary.scalar("Train Loss", train_cost)
            tf.summary.scalar("Validation Loss", validation_cost)
            tf.summary.scalar("Training Accuracy", accuracy)
            tf.summary.scalar("Validation Accuracy", accuracy_valid)

            # merge all summaries into a single "operation" which we can execute in a session 
            summary_op = tf.summary.merge_all()

            saver = tf.train.Saver(max_to_keep=10000)
            
 

        ##########################
        ### TRAINING & EVALUATION
        ##########################

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        with tf.Session(graph=g, config=config) as sess:

            sess.run(tf.global_variables_initializer())
			
            # create log writer object
            writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
            step = 0
            epoch = 0

            validation_loss = 100 
            validation_accuracy = 0

            print "First validation loss: " +str(validation_loss)
            print "First validation accuracy: " +str(validation_accuracy)+ "\n"
            validation_step = step
            num_retries = 0

            train_batch_number = 1
            train_file = 0
            factor = 1.0
            train_acc = 0

            while step < num_steps:

                train_batch_x = np.load(gzip.GzipFile(train_data_dir+'/batch_inputs_'+str(train_file)+'.npy.gz', "r"))
                train_batch_y = np.load(gzip.GzipFile(train_data_dir+'/batch_targets_'+str(train_file)+'.npy.gz', "r"))
                train_input_seq_length = np.load(gzip.GzipFile(train_data_dir+'/batch_input_seq_length_'+str(train_file)+'.npy.gz', "r"))
                train_target_seq_length = np.load(gzip.GzipFile(train_data_dir+'/batch_output_seq_length_'+str(train_file)+'.npy.gz', "r"))
                    
                learning_rate_value, _, loss, train_batch_acc, summary = sess.run([learning_rate, train, train_cost, accuracy, summary_op], feed_dict={
                                                                                   inputs:  train_batch_x,
                                                                                   targets: train_batch_y,
                                                                                   input_seq_length:  train_input_seq_length,
                                                                                   target_seq_length: train_target_seq_length,
                                                                                   learning_rate_factor: factor,
                                                                                   initial_learning_rate: learning_rate,
                                                                                   training_phase: True})


                learning_rate = learning_rate_value
                train_acc += train_batch_acc

                if factor == 0.5:
                    factor = 1.0
                
                #print "Step number: "+ str(step+1) + " Training Batch Number: "+ str(train_file+1)+" Learning Rate: " + str(learning_rate_value)

                train_batch_number = train_batch_number + 1
                train_file = (train_batch_number % training_batch_total) - 1
                if train_file == -1:
                    train_file = training_batch_total - 1

                # write log to display in tensorboard
                writer.add_summary(summary, train_batch_number)

                step = step + 1

                if step % valid_frequency == 0:

                    epoch = train_batch_number / training_batch_total

                    sum_batch_current_loss = 0
                    valid_acc = 0
                    for valid_file in range(valid_batch_total):

                        validation_x = np.load(gzip.GzipFile(valid_data_dir+'/batch_inputs_'+str(valid_file)+'.npy.gz', "r"))
                        validation_y = np.load(gzip.GzipFile(valid_data_dir+'/batch_targets_'+str(valid_file)+'.npy.gz', "r"))
                        validation_x_seq_length = np.load(gzip.GzipFile(valid_data_dir+'/batch_input_seq_length_'+str(valid_file)+'.npy.gz', "r"))
                        validation_y_seq_length = np.load(gzip.GzipFile(valid_data_dir+'/batch_output_seq_length_'+str(valid_file)+'.npy.gz', "r"))

                        loss, validation_batch_acc, summary = sess.run([validation_cost, accuracy_valid, summary_op], feed_dict={
                                                                                                                  inputs:  validation_x,
                                                                                                                  targets: validation_y,
                                                                                              input_seq_length: validation_x_seq_length,
                                                                                             target_seq_length:validation_y_seq_length,
                                                                                             training_phase: False})
                        sum_batch_current_loss += loss
                        valid_acc += validation_batch_acc
                    
					current_loss = sum_batch_current_loss / valid_batch_total
                    valid_acc /= valid_batch_total
					
                    #we will only check rounded 3 decimal points
                    current_validation_accuracy = float(format(valid_acc, '.3f'))

                    train_acc /= (training_batch_total)

                    # writing accuracy information in a log file
                    accuracy_log_file = open(logdir+'/accuracy_log', "a")
                    print "\nEpoch: %03d Train/Valid Accuracy: %.3f/%.3f\n" % (epoch, train_acc, valid_acc)
                    accuracy_log_file.write("Epoch: %03d | Learning Rate: %f | Train/Valid ACC: %.3f/%.3f" % (epoch, learning_rate_value, train_acc, valid_acc)+"\n")
                    accuracy_log_file.close()

                    train_acc = 0

                    #saving weights and biases after each epoch
                    #saver.save(sess, NN_dir + '/model.ckpt', global_step=train_batch_number - 1)
                    
                    if current_loss >= validation_loss or current_validation_accuracy <= validation_accuracy:

                        print "Make learning rate half, Current_loss: "+str(current_loss) + " Validation_loss: " + str(validation_loss)
                        print "Epoch: " + str(epoch)+ " Step number: "+ str(step+1) + " Training Batch Number: "+ str(train_file+1)+" New Learning Rate: " + str(learning_rate_value*.5)

                        factor = 0.5

                        step = validation_step
                        validation_accuracy = current_validation_accuracy
                        num_retries = num_retries + 1
                        print "Number of Retries: " +str(num_retries)+"\n"

                        if num_retries == total_number_of_retries:
       
                            saver.save(sess, NN_dir + '/model.ckpt', global_step=train_batch_number - 1)
                            save_batch_file = open(NN_dir+'/save_batch_number', "w")
                            save_batch_file.write(str(train_batch_number - 1))
                            save_batch_file.close()

                            print "Number of retries reaches maximum, finishing training the model"
                            break

                        continue

                    else:
                        print "Keep learning rate same, Current_loss: "+str(current_loss) + " Validation_loss: " + str(validation_loss)
                        print "Epoch: " + str(epoch)+ " Step number: "+ str(step+1) + " Training Batch Number: "+ str(train_file+1)+" New Learning Rate: " + str(learning_rate_value)+"\n"
                        factor = 1.0
                        validation_loss = current_loss
                        validation_accuracy = current_validation_accuracy
                        validation_step = step

                        num_retries = 0

                if step == num_steps:
                    saver.save(sess, NN_dir + '/model.ckpt', global_step=train_batch_number - 1)
                    save_batch_file = open(NN_dir+'/save_batch_number', "w")
                    save_batch_file.write(str(train_batch_number - 1))
                    save_batch_file.close()
            
