'''@file ark2hdf5.py
reading features and convert to hdf5 formalt'''

import ark
import numpy as np
import readfiles
import h5py
import gzip

class HDF5(object):
    '''Class that can read features from a Kaldi archive and convert to hdf5 formalt'''

    def __init__(self, scpfile, target_coder, target_path, hdf5_path, input_dim):
        '''
        create a FeatureReader object

        Args:
            scpfile: path to the features .scp file
            max_input_length: the maximum length of all the utterances in the
                scp file
        '''

        #create the feature reader
        self.reader = ark.ArkReader(scpfile)
        #get a dictionary connecting training utterances and targets.
        target_dict = {}

        with gzip.open(target_path, 'rb') as fid:
            for line in fid:
                splitline = line.strip().split(' ')
                target_dict[splitline[0]] = ' '.join(splitline[1:])

        self.target_dict = target_dict

        self.hdf5_path = hdf5_path
        #the input dim of network
        self.input_dim = input_dim

        #save the target coder
        self.target_coder = target_coder

        self.ark2hdf5()

    def get_utt(self):
        '''
        read the next features from the archive, normalize and splice them

        Returns:
            the normalized and spliced features
        '''
        
        #read utterance
        (utt_id, utt_mat, looped) = self.reader.read_next_utt()

        return utt_id, utt_mat, looped

    def ark2hdf5(self):
    	'''
    	convert features from kaldi archive to hdf5 formalt

    	'''
    	cols = self.input_dim + 1

    	h5f = h5py.File(self.hdf5_path, 'w')

        dataset = h5f.create_dataset("data", (10240, cols),
                                     maxshape=(None, cols),
                                     chunks=(1024000, cols),
                                     dtype='float32')

        utt_id, utt_mat, looped = self.get_utt()

        if utt_id in self.target_dict and utt_mat is not None:
            target = self.target_dict[utt_id]
            encoded_target = self.target_coder.encode(target)

            len_utt = utt_mat.shape[0]
            len_target = encoded_target.shape[0]
            #check the length of target and frame number

            if abs(len_utt - len_target) <=5:
                min_len = min(len_utt, len_target)
                inputs = utt_mat[0:min_len,]
                targets = encoded_target[0:min_len,]
            else:
                print 'WARNING no targets for %s' % utt_id
        else:
            if utt_id not in self.target_dict:
                print 'WARNING no targets for %s' % utt_id
            if utt_mat is None:
                print 'WARNING %s is too short to splice' % utt_id

        times = 0
        while(not looped):
            utt_id, utt_mat, looped = self.get_utt()
            if inputs.shape[0] < 1024000:
                if utt_id in self.target_dict and utt_mat is not None:
                    target = self.target_dict[utt_id]
                    encoded_target = self.target_coder.encode(target)

                    len_utt = utt_mat.shape[0]
                    len_target = encoded_target.shape[0]
                    #check the length of target and frame number

                    if abs(len_utt - len_target) <=5:
                        min_len = min(len_utt, len_target)
                        inputs = np.row_stack((inputs, utt_mat[0:min_len,]))
                        targets = np.append(targets,encoded_target[0:min_len,])
                    else:
                        print 'WARNING no targets for %s' % utt_id
                else:
                   if utt_id not in self.target_dict:
                        print 'WARNING no targets for %s' % utt_id
                   if utt_mat is None:
                        print 'WARNING %s is too short to splice' % utt_id

                if looped:
                    feat_and_target = np.column_stack((inputs, targets))
                    np.random.shuffle(feat_and_target)

                    append = 10240 - (feat_and_target.shape[0] % 10240)
                    feat_and_target = np.row_stack((feat_and_target, np.zeros((append, cols))))
                    print feat_and_target.shape
                    
                    for i in range(feat_and_target.shape[0] / 10240):
                        dataset.resize([times*10240+10240, cols])
                        dataset[times*10240:times*10240+10240] = feat_and_target[i*10240:i*10240+10240,:]
                        times += 1


            else:
            	print "======================================"
            	print inputs.shape
            	print targets.shape
                feat_and_target = np.column_stack((inputs, targets))
                np.random.shuffle(feat_and_target)
                print feat_and_target.shape

                for i in range(100):
                    dataset.resize([times*10240+10240, cols])
                    dataset[times*10240:times*10240+10240] = feat_and_target[i*10240:i*10240+10240,:]
                    times += 1
                targets = feat_and_target[1024000:inputs.shape[0],inputs.shape[1]:inputs.shape[1]+1]
                inputs =  feat_and_target[1024000:inputs.shape[0],0:inputs.shape[1]]
                
                print targets.shape
                print inputs.shape
                np.reshape(targets, (inputs.shape[0] - 1024000, 1))
                if utt_id in self.target_dict and utt_mat is not None:
                    target = self.target_dict[utt_id]
                    encoded_target = self.target_coder.encode(target)

                    len_utt = utt_mat.shape[0]
                    len_target = encoded_target.shape[0]
            		#check the length of target and frame number

                    if abs(len_utt - len_target) <=5:
                        min_len = min(len_utt, len_target)
                        inputs = np.row_stack((inputs, utt_mat[0:min_len,]))
                        targets = np.append(targets,encoded_target[0:min_len,])
                    else:
                        print 'WARNING no targets for %s' % utt_id
                else:
                    if utt_id not in self.target_dict:
                        print 'WARNING no targets for %s' % utt_id
                    if utt_mat is None:
                        print 'WARNING %s is too short to splice' % utt_id


    	
        h5f.close()
