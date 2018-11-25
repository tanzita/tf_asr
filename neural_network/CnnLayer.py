'''
@author: Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@date: 21.12.2017
@version: 1.0+
'''
import tensorflow as tf
import numpy as np

class CnnLayer(object):

    def __init__(self):
	
        print 'Init cnn layer'

   
    def __call__(self, inputs, scope=None):
    
        with tf.variable_scope(scope or type(self).__name__):
            
            print 'Layer: ' + scope
            print 'Input: ' 
            print inputs.get_shape()

            with tf.variable_scope('prep_data_l1'):             

                inputs_img = tf.reshape(inputs, tf.stack( [ tf.shape(inputs)[0] , 7, 3, 13] )  )
                inputs_img = tf.transpose(inputs_img, [ 0 , 1, 3, 2 ] )  
    
            print 'Input Img: ' 
            print inputs_img.get_shape().as_list()

            hidden = self.convolution(inputs_img, 'conv_l1', 3, 256, 3, 3)

            with tf.variable_scope('pool_l1'):
			
                pool = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 3, 1], padding='VALID')

            print 'poll_l1: '
            print pool.get_shape().as_list()

            hidden = self.convolution(pool, 'conv_l2', 256, 256, 3, 4)
            
            with tf.variable_scope('out_op'):
			
                shape = hidden.get_shape().as_list()
                outputs = tf.reshape(hidden, tf.stack( [tf.shape(hidden)[0], shape[1]  * shape[2]  * shape[3]   ] ) )
         
            print 'Outputs: ' 
            print outputs.get_shape().as_list()
        
        return outputs

    def convolution(self, inputs_img, name_layer, in_dim, out_dim, t_conv_size, f_conv_size):
	
        with tf.variable_scope('parameters_'+name_layer):
		
            n = t_conv_size*f_conv_size*out_dim
            weights = tf.get_variable('weights_'+name_layer, [t_conv_size, f_conv_size, in_dim, out_dim],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            biases = tf.get_variable('biases_'+name_layer,   [out_dim],   initializer=tf.constant_initializer(0) )

        with tf.variable_scope('conv_'+name_layer):
		
            conv = tf.nn.conv2d(inputs_img,  weights, [1, 1, 1, 1], padding='VALID')
            #print conv.get_shape()
            conv = tf.contrib.layers.batch_norm(conv,
                scope='batch_norm')
            hidden = tf.nn.relu(conv + biases)

            print 'hidden_'+ name_layer
            print hidden.get_shape()

        return hidden
