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
        self.save_dir = config.get('directories', 'exp_dir') + '/train_features_dir_diff_batch'
        self.batch_size = [15, 16, 17, 18, 19] 
        self.pdf_file_dir = config.get('directories', 'train_pdf_dir')
        self.pdf_file_total = int(config.get('general', 'train_num_pdf_files'))
        self.ark_file = config.get('general', 'train_ark')
        self.train_cmvn_ark = config.get('general', 'train_cmvn_ark')
        self.cmvn_dict = self.get_cmvn_dict(self.train_cmvn_ark)

        self.max_length = 0
        self.total_number_of_utterances = 0
        self.input_dim = 0
        self.input_dim_check = False


        utt_2_spk_file = config.get('general', 'train_utt_2_spk')
        self.utt_2_spk_dict = self.get_utt_2_spk_dict(utt_2_spk_file)

        self.num_labels = int(config.get('general', 'train_num_label'))

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
            too short to splice None will be returned
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

    def make_target_dict(self):
        '''
        read the file containing the state alignments

        Args:
            target_path: path to the alignment file

        Returns:
            A dictionary containing
                - Key: Utterance ID
                - Value: The state alignments as a space seperated string
        '''

        #put all the alignments in one file
        all_ali_files = [self.pdf_file_dir + '/pdf.' + str(i+1) + '.gz' for i in range(self.pdf_file_total)]
       
        target_dict = {}

        for zip_file in all_ali_files:
            zip_file = zip_file.replace("\n", "") # commented by tanzi
	    with gzip.open(zip_file, 'rb') as fid:
                for line in fid:
                    splitline = line.strip().split(' ')
                    target_dict[splitline[0]] = ' '.join(splitline[1:])

        return target_dict

    def save_target_prior(self):
        '''
        compute the count of the targets in the data

        Returns:
            a numpy array containing the counts of the targets
        '''

        self.max_target_length = 0
        target_array_list = []

        for target_string in self.target_dict.values():
            target_list = target_string.split(' ')
            if self.max_target_length < len(target_list):
                self.max_target_length = len(target_list)
            target_array = np.array(target_list, dtype=np.uint32)
            target_array_list.append(target_array)

        #create a big vector of stacked targets
        all_targets = np.concatenate(target_array_list)

        #count the number of occurences of each target
        prior = np.bincount(all_targets, minlength=self.num_labels)

        prior = prior.astype(np.float32)

        prior = prior/prior.sum()

        np.save(self.pdf_file_dir + '/prior.npy', prior)


    def get_target_array(self, utt_id):

        target_sequence = self.target_dict[utt_id]
        target_sequence_string_list = target_sequence.strip().split(' ')
        targets_list = [int(i) for i in target_sequence_string_list]
        targets = np.array(targets_list)
 
        return targets

    def get_utterance_dict(self, file_name):

        seq_length_count = 0
        target_match = False
        utt_id = ""
        utt_dict = {}

        raw_data = open(file_name)
        file_data = raw_data.read().split("\n")
        raw_data.close()

        for line in file_data:
            list_line = line.split()

            if len(list_line) > 0:
  
                if list_line[1] == "[" and list_line[0] in self.target_dict:

                    target_match = True
                    utt_id = list_line[0]
                    features_per_frame = []
                    

                elif list_line[-1] == "]" and target_match:
    
                    del list_line[-1]
                    seq_length_count += 1

                    fetures_list = [float(i) for i in list_line]
                    features_per_frame.append(fetures_list)
                    fetures_per_utt = np.array(features_per_frame)
                    fetures_per_utt_normalized = self.apply_cmvn(fetures_per_utt, utt_id)
                    splice_fetures_per_utt, splice_done = self.splice(fetures_per_utt_normalized)

                    if splice_done:

                        utt_dict[utt_id] = splice_fetures_per_utt

                        target_match = False

                        if seq_length_count > self.max_length:
                            self.max_length = seq_length_count

                        seq_length_count = 0
                        self.total_number_of_utterances += 1
                        print 'utt_no: ' + str(self.total_number_of_utterances) + ' utt_id: ' + utt_id

                        #get the input dim number only once
                        if self.input_dim_check == False:
                            self.input_dim = splice_fetures_per_utt.shape[1]
                            self.input_dim_check = True

                    else:
                        seq_length_count = 0


                elif target_match:

                    fetures_list = [float(i) for i in list_line]
                    features_per_frame.append(fetures_list)
                    seq_length_count += 1


        return utt_dict

    def save_batch_data(self, kaldi_batch_data, kaldi_batch_labels, batch_size, step):
        
        processed_batch_inputs, processed_batch_targets, processed_batch_input_seq_length, processed_batch_output_seq_length =self.process_kaldi_batch_data(kaldi_batch_data, kaldi_batch_labels)


        batch_inputs = gzip.GzipFile(self.save_dir+'/batch_inputs_'+str(batch_size)+"_"+str(step)+'.npy.gz', 'w')
        np.save(batch_inputs, processed_batch_inputs)
        batch_inputs.close()
		
        batch_targets = gzip.GzipFile(self.save_dir+'/batch_targets_'+str(batch_size)+"_"+str(step)+'.npy.gz', 'w')
        np.save(batch_targets, processed_batch_targets)
        batch_targets.close()
		
        batch_input_seq_length = gzip.GzipFile(self.save_dir+'/batch_input_seq_length_'+str(batch_size)+"_"+str(step)+'.npy.gz', 'w')
        np.save(batch_input_seq_length, processed_batch_input_seq_length)
        batch_input_seq_length.close()
		
        batch_output_seq_length = gzip.GzipFile(self.save_dir+'/batch_output_seq_length_'+str(batch_size)+"_"+str(step)+'.npy.gz', 'w')
        np.save(batch_output_seq_length, processed_batch_output_seq_length)
        batch_output_seq_length.close()
        


        print 'Zero padding on Batch ' + str(step + 1) + ' is completed.'


    def process_kaldi_batch_data(self, inputs, targets):
        
        #get a list of sequence lengths
        input_seq_length = [i.shape[0] for i in inputs]
        output_seq_length = [t.shape[0] for t in targets]

        #pad all the inputs qnd targets to the max_length and put them in
        #one array
        padded_inputs = np.array([np.append(
            i, np.zeros([self.max_length-i.shape[0], i.shape[1]]), 0)
                                  for i in inputs])

        padded_targets = np.array([np.append(
            t, np.zeros(self.max_length-t.shape[0]), 0)
                                   for t in targets])

        #transpose the inputs and targets so they fit in the placeholders
        batch_inputs = padded_inputs.transpose([1, 0, 2])
        batch_targets = padded_targets.transpose()

        batch_targets = batch_targets[:, :, np.newaxis]

        return batch_inputs, batch_targets, input_seq_length, output_seq_length


    def batch_data_processing(self):

        self.target_dict = self.make_target_dict()

        self.save_target_prior()
        
        utt_dict = self.get_utterance_dict(self.ark_file)

        #with open(self.save_dir+"/utt_dict", "wb") as fp:
            #pickle.dump(utt_dict, fp)
        
        #with open(self.save_dir+"/utt_dict", "rb") as fp:
            #utt_dict = pickle.load(fp)

        utt_id_list = utt_dict.keys()
        random.shuffle(utt_id_list)

        #print len(utt_id_list)

        utt_mat = []
        target_mat = []
        batch_count = 0

        
        for batch_size in self.batch_size:
            batch_count = 0
            for id_count in range(len(utt_id_list)):

                utt_key = utt_id_list[id_count]
                utt_array = utt_dict[utt_key]
                target_array = self.get_target_array(utt_key)
                utt_mat.append(utt_array)
                target_mat.append(target_array)

                if (id_count + 1) % batch_size == 0:
 
                    self.save_batch_data(utt_mat, target_mat, batch_size, batch_count)
                    utt_mat = []
                    target_mat = []
                    batch_count += 1
            
                if batch_count == 5:
                    break
            

        important_info = {'train_utt_max_length': self.max_length, 
                   'training_batch_total': batch_count - 1, 
                   'total_training_utterances': self.total_number_of_utterances, 
                   'input_dim': self.input_dim,
                   'num_labels':self.num_labels,
                   'train_label_max_length':self.max_target_length}

        with open(self.save_dir+"/train_important_info", "wb") as fp:
            pickle.dump(important_info, fp)

        return important_info
        

def get_important_info(save_dir):

    with open(save_dir+"/train_important_info", "rb") as fp:
        important_info = pickle.load(fp)

    return important_info
