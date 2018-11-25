'''
@author: Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@date: 21.09.2017
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Tanzia Haque Tanzi (tanzita@ims.uni-stuttgart.de)
@license : MIT License
'''
import numpy as np
import gzip
import pickle
import random
import os

class features_extraction(object):

    def __init__(self, config):

        self.context_width = int(config.get('simple_NN', 'context_width'))
        self.ark_file_name = config.get('general', 'test_ark')
        self.save_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
        self.test_cmvn_ark = config.get('general', 'test_cmvn_ark')
        self.cmvn_dict = self.get_cmvn_dict(self.test_cmvn_ark)

        utt_2_spk_file = config.get('general', 'test_utt_2_spk')
        self.utt_2_spk_dict = self.get_utt_2_spk_dict(utt_2_spk_file)

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def get_utt_2_spk_dict(self, file_name):

        utt_2_spk_dict = {}

        raw_data = open(file_name)
        file_data = raw_data.read().split("\n")
        raw_data.close()

        for line in file_data:
            list_line = line.split()
            if list_line:
                utt_2_spk_dict[list_line[0]] = list_line[1]

        return utt_2_spk_dict

    def get_cmvn_dict(self, file_name):

        spk_id = ""
        cmvn_dict = {}

        raw_data = open(file_name)
        file_data = raw_data.read().split("\n")
        raw_data.close()

        for line in file_data:
            list_line = line.split()

            if len(list_line) > 0:
  
                if list_line[1] == "[":

                    spk_id = list_line[0]
                    cmvn_data_list = []
                    
                elif list_line[-1] == "]":
    
                    del list_line[-1]

                    cmvn_data = [float(i) for i in list_line]
                    cmvn_data_list.append(cmvn_data)
                    cmvn_data_array = np.array(cmvn_data_list)
                    cmvn_dict[spk_id] = cmvn_data_array

                else:

                    cmvn_data = [float(i) for i in list_line]
                    cmvn_data_list.append(cmvn_data)

        return cmvn_dict

    def apply_cmvn(self, utt, utt_id):

        '''
        apply mean and variance normalisation

        The mean and variance statistics are computed on previously seen data

        Args:
            utt: the utterance feature numpy matrix
            stats: a numpy array containing the mean and variance statistics. The
                first row contains the sum of all the fautures and as a last element
                the total number of features. The second row contains the squared
                sum of the features and a zero at the end

        Returns:
            a numpy array containing the mean and variance normalized features
        '''

        #spk_id = utt_id[:3]
        spk_id = self.utt_2_spk_dict[utt_id]
    
        stats = self.cmvn_dict[spk_id]

        #compute mean
        mean = stats[0, :-1]/stats[0, -1]

        #compute variance
        variance = stats[1, :-1]/stats[0, -1] - np.square(mean)

        #return mean and variance normalised utterance
        return np.divide(np.subtract(utt, mean), np.sqrt(variance))

    def splice(self, utt):
        '''
        splice the utterance

        Args:
            utt: numpy matrix containing the utterance features to be spliced
            context_width: how many frames to the left and right should
                be concatenated

        Returns:
            a numpy array containing the spliced features, if the features are
            too short to splice then the original utterance will be returned
        '''

        context_width = self.context_width

        #return None if utterance is too short
        if utt.shape[0]<1+2*context_width:
            return utt, False

        #create spliced utterance holder
        utt_spliced = np.zeros(
        shape=[utt.shape[0], utt.shape[1]*(1+2*context_width)],
        dtype=np.float32)

        #middle part is just the uttarnce
        utt_spliced[:, context_width*utt.shape[1]:
                (context_width+1)*utt.shape[1]] = utt

        for i in range(context_width):

            #add left context
            utt_spliced[i+1:utt_spliced.shape[0],
                    (context_width-i-1)*utt.shape[1]:
                    (context_width-i)*utt.shape[1]] = utt[0:utt.shape[0]-i-1, :]

            #add right context
            utt_spliced[0:utt_spliced.shape[0]-i-1,
                    (context_width+i+1)*utt.shape[1]:
                    (context_width+i+2)*utt.shape[1]] = utt[i+1:utt.shape[0], :]

        return utt_spliced, True

    def data_processing(self):
        
        utt_dict = {}

        raw_data = open(self.ark_file_name)
        file_data = raw_data.read().split("\n")
        raw_data.close()

        utt_id = ""
        seq_length_count = 0
        max_length = 0
        total_number_of_utterances = 0

        for line in file_data:
            list_line = line.split()

            if len(list_line) > 0:

                if list_line[1] == "[":

                        utt_id = list_line[0]
                        features_per_frame = []
 
                elif list_line[-1] == "]":
    
                    del list_line[-1]

                    seq_length_count += 1

                    fetures_list = [float(i) for i in list_line]
                    features_per_frame.append(fetures_list)
                    fetures_per_utt = np.array(features_per_frame)
                    fetures_per_utt_normalized = self.apply_cmvn(fetures_per_utt, utt_id)
                    splice_fetures_per_utt, splice_done = self.splice(fetures_per_utt_normalized)

                    if splice_done:

                        utt_dict[utt_id] = splice_fetures_per_utt
                        #print total_number_of_utterances + " " + utt_id
                        
                        if seq_length_count > max_length:
                            max_length = seq_length_count

                        seq_length_count = 0
                        total_number_of_utterances += 1
                        print str(total_number_of_utterances) + " " + utt_id

                    else:

                        seq_length_count = 0

                else:

                    fetures_list = [float(i) for i in list_line]
                    features_per_frame.append(fetures_list)
                    seq_length_count += 1

        with open(self.save_dir+"/utt_dict", "wb") as fp:
            pickle.dump(utt_dict, fp)

        important_info = {'test_utt_max_length': max_length,  
                   'total_test_utterances': total_number_of_utterances}

        with open(self.save_dir+"/test_important_info", "wb") as fp:
            pickle.dump(important_info, fp)

        return important_info


def get_important_info(save_dir):

    with open(save_dir+"/test_important_info", "rb") as fp:
        important_info = pickle.load(fp)

    return important_info


