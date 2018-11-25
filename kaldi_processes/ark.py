'''
@file ark.py
contains the .ark io functionality

Copyright 2014    Yajie Miao    Carnegie Mellon University
           2015    Yun Wang      Carnegie Mellon University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
MERCHANTABLITY OR NON-INFRINGEMENT.
See the Apache 2 License for the specific language governing permissions and
limitations under the License.
'''

import struct
import numpy as np

np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.nan)


class ArkWriter(object):
    '''
    Class to write numpy matrices into Kaldi .ark file and create the
    corresponding .scp file. It only supports binary-formatted .ark files. Text
    and compressed .ark files are not supported. The inspiration for this class
    came from pdnn toolkit (see licence at the top of this file)
    (https://github.com/yajiemiao/pdnn)
    '''

    def __init__(self, scp_path, default_ark):
        '''
        Arkwriter constructor

        Args:
            scp_path: path to the .scp file that will be written
            default_ark: the name of the default ark file (used when not
                specified)
        '''

        self.scp_path = scp_path
        self.scp_file_write = open(self.scp_path, 'w')
        self.default_ark = default_ark

    def write_next_utt(self, utt_id, utt_mat, ark_path=None):
        '''
        read an utterance to the archive

        Args:
            ark_path: path to the .ark file that will be used for writing
            utt_id: the utterance ID
            utt_mat: a numpy array containing the utterance data
        '''

        ark = ark_path or self.default_ark
        ark_file_write = open(ark, 'ab')
        utt_mat = np.asarray(utt_mat, dtype=np.float32)
        rows, cols = utt_mat.shape
        ark_file_write.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
        pos = ark_file_write.tell()
        ark_file_write.write(struct.pack('<xcccc', 'B', 'F', 'M', ' '))
        ark_file_write.write(struct.pack('<bi', 4, rows))
        ark_file_write.write(struct.pack('<bi', 4, cols))
        ark_file_write.write(utt_mat)
        self.scp_file_write.write('%s %s:%s\n' % (utt_id, ark, pos))
        ark_file_write.close()

    def close(self):
        '''close the ark writer'''

        self.scp_file_write.close()
