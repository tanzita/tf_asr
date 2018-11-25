'''
@author: Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@date: 22.12.2017
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@license : MIT License
'''

import tensorflow as tf
import numpy as np
import seq_convertors
import pickle
import os

class Decode(object):

    def __init__(self, config, train_important_information, test_important_information):

        decode_dir = config.get('directories', 'exp_dir') + '/CNN_decode_dir'
        self.load_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
        self.save_dir = config.get('directories', 'exp_dir')  + '/CNN_train_dir'
        self.max_length = test_important_information['test_utt_max_length']
        self.input_dim = train_important_information['input_dim']
        self.prior = np.load(config.get('directories', 'train_pdf_dir') + '/prior.npy')
        # To avoiding infinity if prior value is zero for pdf-id 'n' with no state
        self.prior =  np.where(self.prior == 0, np.finfo(float).eps, self.prior)
        self.total_uttarences = test_important_information['total_test_utterances']
        self.epochs = config.get('simple_NN', 'training_epochs')
        self.hid_layer_num = int(config.get('simple_NN', 'hidden_layer_num'))
        utt_file_name = config.get('general', 'test_utt_2_spk')

        with open(self.load_dir + "/utt_dict", "rb") as fp:
            self.utt_dict = pickle.load(fp)

        #self.utt_id_list = self.utt_dict.keys()
        self.utt_id_list = self.get_list(utt_file_name)

        self.decode_dir = config.get('directories', 'exp_dir') + '/CNN_decode_dir'
        if not os.path.isdir(self.decode_dir):
            os.mkdir(self.decode_dir)


    def convolution(self, inputs_img, weights, biases):
        
        conv = tf.nn.conv2d(inputs_img,  weights, [1, 1, 1, 1], padding='VALID')
        #print conv.get_shape()

        hidden = tf.nn.relu(conv + biases)
        #print 'hidden_'+ name_layer
        print 'hidden_'
        print hidden.get_shape()

        return hidden


    def get_list(self, utt_file_name):

        utt_id_list = []

        raw_data = open(utt_file_name)
        utt_file_data = raw_data.read().split("\n")
        raw_data.close()


        for index in range(len(utt_file_data)):

            list_utt = utt_file_data[index].split()

            if list_utt:
	        utt_id_list.append(list_utt[0])

        return utt_id_list


    def retrieved_data(self):

        raw_data = open(self.save_dir+"/save_batch_number")
        save_file_id = str(raw_data.read())#.split("\n")
        raw_data.close()

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9


        with tf.Session(config=config) as sess:

        #with tf.Session() as sess:

            saver = tf.train.import_meta_graph(self.save_dir+'/'+'model.ckpt-'+save_file_id+'.meta')
            
            saver.restore(sess, self.save_dir+'/'+'model.ckpt-'+save_file_id)

            #variable_list = tf.global_variables()

            #print variable_list
            
            self.conv1_weights = sess.run("weights_conv_l1:0")
            print self.conv1_weights.shape
            self.conv1_biases = sess.run("biases_conv_l1:0")
            print self.conv1_biases.shape

            self.conv2_weights = sess.run("weights_conv_l2:0")
            print self.conv2_weights.shape
            self.conv2_biases = sess.run("biases_conv_l2:0")
            print self.conv2_biases.shape

            self.weights = {'h'+str(i): sess.run("weights/h"+str(i)+"_value:0") for i in range(1,self.hid_layer_num + 1)}
            self.weights['out'] = sess.run("weights/weight_out_value:0")

            self.biases = {'b'+str(i): sess.run("biases/b"+str(i)+"_value:0") for i in range(1,self.hid_layer_num + 1)}
            self.biases['out'] = sess.run("biases/bias_out_value:0")




    def decode_data(self, writer):

        self.retrieved_data()

        ##########################
        ### GRAPH DEFINITION
        ##########################

        g = tf.Graph()
        with g.as_default():

            decode_inputs = tf.placeholder(
                         tf.float32, shape=[self.max_length, self.input_dim], name='inputs')

            decode_seq_length = tf.placeholder(
                                  tf.int32, shape=[1], name='seq_length')

            split_inputs = tf.unstack(tf.expand_dims(decode_inputs, 1),
                                                            name="decode_split_inputs_op")

            nonseq_inputs = seq_convertors.seq2nonseq(split_inputs, decode_seq_length)

            inputs_img = tf.reshape(nonseq_inputs, tf.stack( [ tf.shape(nonseq_inputs)[0] , 7, 1, 13] )  )
            inputs_img = tf.transpose(inputs_img, [ 0 , 1, 3, 2 ] )

            print 'Input Img: '
            print inputs_img.get_shape().as_list()

            hidden_1 = self.convolution(inputs_img, self.conv1_weights, self.conv1_biases)

            
            pool = tf.nn.max_pool(hidden_1, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

            print 'poll_l1: '
            print pool.get_shape().as_list()

            hidden_2 = self.convolution(pool, self.conv2_weights, self.conv2_biases)

            
            shape = hidden_2.get_shape().as_list()
            conv_outputs = tf.reshape(hidden_2, tf.stack( [tf.shape(hidden_2)[0], shape[1]  * shape[2]  * shape[3]   ] ) )

            print 'Outputs: '
            print conv_outputs.get_shape().as_list()

            
            # Multilayer perceptron

            #-----------Start-----------

            layer_1 = tf.add(tf.matmul(conv_outputs, self.weights['h1']), self.biases['b1'])

            layer_out = tf.nn.tanh(layer_1)

            for i in range(2,self.hid_layer_num+1):

                layer = tf.add(tf.matmul(layer_out, self.weights['h'+str(i)]), self.biases['b'+str(i)])

                layer_out = tf.nn.tanh(layer)


            logits = tf.add(tf.matmul(layer_out, self.weights['out']), self.biases['out'])

            outputs = tf.nn.softmax(logits, name="final_operation")

            #--------- End--------------------------


        ##########################
        ###      EVALUATION
        ##########################

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9


        with tf.Session(graph=g, config=config) as sess:

        #with tf.Session(graph=g) as sess:

            sess.run(tf.global_variables_initializer())

            for i in range(self.total_uttarences):

                utt_id = self.utt_id_list[i]


                utt_mat = self.utt_dict[utt_id]

                input_seq_length = [utt_mat.shape[0]]

                #print "This is the input length: "+str(input_seq_length)
                
                #pad the inputs
                utt_mat = np.append( utt_mat, np.zeros([self.max_length-utt_mat.shape[0], 
                                                                          utt_mat.shape[1]]), 0)
                
                outputs_value = sess.run(outputs, feed_dict={decode_inputs: utt_mat,
                                                       decode_seq_length: input_seq_length})

 
                print str(i+1)+" "+str(self.total_uttarences)+" "+str(utt_id)+" "+str(outputs_value.shape)

                #get state likelihoods by dividing by the prior
                output = outputs_value/self.prior

                #floor the values to avoid problems with log
                output = np.where(output == 0, np.finfo(float).eps, output)

                # print (output.shape)
                # print (type(output))

                #write the pseudo-likelihoods in kaldi feature format
                writer.write_next_utt(utt_id, np.log(output))

        #close the writer
        writer.close()
