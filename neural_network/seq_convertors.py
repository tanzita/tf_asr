'''
The MIT License (MIT)

Copyright (c) 2016 Vincent Renkens

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


'''@file seq_convertors.py
this file contains functions that convert sequential data to non-sequential data
and the other way around. Sequential data is defined to be data that is suetable
as RNN input. This means that the data is a list containing an N x F tensor for
each time step where N is the batch size and F is the input dimension non
sequential data is data suetable for input to fully connected layers. This means
that the data is a TxF tensor where T is the sum of all sequence lengths. This
functionality only works for q specified batch size'''

import tensorflow as tf

def seq2nonseq(tensorlist, seq_length, name=None):
    '''
    Convert sequential data to non sequential data

    Args:
        tensorlist: the sequential data, wich is a list containing an N x F
            tensor for each time step where N is the batch size and F is the
            input/output dimension
        seq_length: a vector containing the sequence lengths
        name: [optional] the name of the operation

    Returns:
        non sequential data, which is a TxF tensor where T is the sum of all
        sequence lengths
    '''

    with tf.name_scope(name or 'seq2nonseq'):
        #convert the list for each time step to a list for each sequence
        sequences = tf.unstack(tf.stack(tensorlist), axis=1)

        #remove the padding from sequences
        sequences = [tf.gather(sequences[s], tf.range(seq_length[s]))
                     for s in range(len(sequences))]

        #concatenate the sequences
        #tensor = tf.concat(0, sequences)
	tensor = tf.concat(sequences, 0)

    return tensor

def nonseq2seq(tensor, seq_length, length, name=None):
    '''
    Convert non sequential data to sequential data

    Args:
        tensor: non sequential data, which is a TxF tensor where T is the sum of
            all sequence lengths
        seq_length: a vector containing the sequence lengths
        length: the constant length of the output sequences
        name: [optional] the name of the operation

    Returns:
        sequential data, wich is a list containing an N x F
        tensor for each time step where N is the batch size and F is the
        input/output dimension
    '''

    with tf.name_scope(name or'nonseq2seq'):
        #get the cumulated sequence lengths to specify the positions in tensor
        #cum_seq_length = tf.concat(0, [tf.constant([0]), tf.cumsum(seq_length)])
        cum_seq_length = tf.concat([tf.constant([0]), tf.cumsum(seq_length)], 0)

        #get the indices in the tensor for each sequence
        indices = [tf.range(cum_seq_length[l], cum_seq_length[l+1])
                   for l in range(int(seq_length.get_shape()[0]))]

        #create the non-padded sequences
        sequences = [tf.gather(tensor, i) for i in indices]

        #pad the sequences with zeros
        sequences = [tf.pad(sequences[s], [[0, length-seq_length[s]], [0, 0]])
                     for s in range(len(sequences))]

        #specify that the sequences have been padded to the constant length
        for seq in sequences:
            seq.set_shape([length, int(tensor.get_shape()[1])])

        #convert the list for eqch sequence to a list for eqch time step
        tensorlist = tf.unstack(tf.stack(sequences), axis=1)

    return tensorlist
