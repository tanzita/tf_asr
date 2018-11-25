'''
@author: Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@license : MIT License
'''

'''@file main.py
run this file to go through the neural net training procedure, look at the 'config.cfg' file to modify the settings'''

import os
from six.moves import configparser
from feature_extraction import train_features_extraction, test_features_extraction, valid_features_extraction
from neural_network import simple_NN_training, simple_NN_decoding
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
config.read('config.cfg')

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

    trainer = simple_NN_training.Simple_multy_layer_perceptron()
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

    if not test_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
        test_important_information = test_features_extraction.get_important_info(save_dir)
            
    decoder = simple_NN_decoding.Decode(config, train_important_information, test_important_information)

    decode_dir = config.get('directories', 'exp_dir') + '/NN_decode_dir'
    #create an ark writer for the likelihoods
    if os.path.isfile(decode_dir + '/likelihoods.ark'):
        os.remove(decode_dir + '/likelihoods.ark')
    
    writer = ark.ArkWriter(decode_dir + '/feats.scp', decode_dir + '/likelihoods.ark')

    decoder.decode_data(writer)

    print "Neural network decoding is completed"

if DECODE_KALDI:

    print '------- decoding testing sets using kaldi decoder ----------'

    decode_dir = config.get('directories', 'exp_dir') + '/NN_decode_dir'

    #copy the gmm model and some files to speaker mapping to the decoding dir
    os.system('cp %s %s' %(config.get('directories', 'train_pdf_dir' ) + '/final.mdl', decode_dir))
    os.system('cp -r %s %s' %(config.get('directories', 'train_graph_dir'), decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/utt2spk', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/text', decode_dir))

    #change directory to kaldi egs
    os.chdir(config.get('directories', 'kaldi_egs'))

    #decode using kaldi
    os.system('/mount/arbeitsdaten/asr/tanzita/kaldi-trunk/egs/ami/ami_test/steps/decode_fmllr.sh --cmd %s --nj %s %s/graph_ami_web_fsh_swbd_web_sw_web_mtg_opensub.o3g.kn.pr1-9 %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % (config.get('general', 'cmd'), config.get('general', 'num_jobs'), decode_dir, decode_dir, decode_dir, decode_dir))
   

    #go back to working dir
    os.chdir(current_dir)

    print "Kaldi decoding is completed"
