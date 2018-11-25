'''
@author: Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@license : MIT License
'''

'''@file msin_CNN.py
run this file to go through the neural net training procedure, look at the 'config.cfg' file to modify the settings'''

import os
from six.moves import configparser
from feature_extraction import train_features_extraction, test_features_extraction, valid_features_extraction
from neural_network import CNN_baseline, CNN_decoding_all_layer 
from kaldi_processes import ark
import numpy as np
import pickle

#select which one to run by providing 'True'. It is arranged sequentially and independently. So, no need to re-run any module succefully more than once

# All modules
TRAIN_FEATURE_EXTRACTION = True
VALID_FEATURE_EXTRACTION = True 
TRAIN_NN = True
TEST_FEATURE_EXTRACTION = True
DECODE_NN = True
DECODE_KALDI = False

#read config file
config = configparser.ConfigParser()
config.read('configuration_mfcc.cfg')

current_dir = os.getcwd()

train_important_information = {}
valid_important_information = {}
test_important_information = {}

if TRAIN_FEATURE_EXTRACTION:
   
    train_features = train_features_extraction.features_extraction(config)

    train_important_information = train_features.batch_data_processing()

    print train_important_information

    print "Train feature extraction is completed."

if VALID_FEATURE_EXTRACTION:

    if not train_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/train_features_dir'
        train_important_information = train_features_extraction.get_important_info(save_dir)

    valid_features = valid_features_extraction.features_extraction(config, train_important_information)

    valid_important_information = valid_features.batch_data_processing()

    print valid_important_information

    print "Valid feature extraction is completed."


if TRAIN_NN:
    
    if not train_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/train_features_dir'
        train_important_information = train_features_extraction.get_important_info(save_dir)

    print train_important_information
		
    
    if not valid_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/valid_features_dir'
        valid_important_information = valid_features_extraction.get_important_info(save_dir)

    print valid_important_information

    trainer = CNN_baseline.Vanila_conv_net()
    trainer.train_NN(config, train_important_information, valid_important_information)

    print "Neural network training is completed"


if TEST_FEATURE_EXTRACTION:

    test_features = test_features_extraction.features_extraction(config)

    test_important_information = test_features.data_processing()

    print test_important_information

    print "Test feature extraction is completed."


if DECODE_NN:

    if not train_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/train_features_dir'
        train_important_information = train_features_extraction.get_important_info(save_dir)
    print train_important_information

    if not test_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
        test_important_information = test_features_extraction.get_important_info(save_dir)
    print test_important_information

    decoder = CNN_decoding_all_layer.Decode(config, train_important_information, test_important_information)

    decode_dir = config.get('directories', 'exp_dir') + '/CNN_decode_dir'
    #create an ark writer for the likelihoods
    if os.path.isfile(decode_dir + '/likelihoods.ark'):
        os.remove(decode_dir + '/likelihoods.ark')
    
    writer = ark.ArkWriter(decode_dir + '/feats.scp', decode_dir + '/likelihoods.ark')

    decoder.decode_data(writer)

    print "Neural network decoding is completed"

if DECODE_KALDI:

    print '------- decoding testing sets using kaldi decoder ----------'

    decode_dir = config.get('directories', 'exp_dir') + '/CNN_decode_dir'

    #copy the gmm model and some files to speaker mapping to the decoding dir
    os.system('cp %s %s' %(config.get('directories', 'train_dir' ) + '/final.mdl', decode_dir))
    os.system('cp -r %s %s' %(config.get('directories', 'train_graph_dir'), decode_dir+"/graph"))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/utt2spk', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/text', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/glm', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/reco2file_and_channel', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/segments', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/spk2utt', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/stm', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/wav.scp', decode_dir))

    #change directory to kaldi egs
    os.chdir(config.get('directories', 'kaldi_egs'))

    #decode using kaldi
    os.system('%s/kaldi_processes/decode_modified.sh --cmd %s --nj %s --mic %s %s/graph %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % (current_dir, config.get('general', 'cmd'), config.get('general', 'num_jobs'), config.get('general', 'mic'), decode_dir, decode_dir, decode_dir, decode_dir))
   

    #go back to working dir
    os.chdir(current_dir)

    print "Kaldi decoding is completed"
 
