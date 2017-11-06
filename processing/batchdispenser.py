'''
#@package batchdispenser
# contain the functionality for read features and batches
# of features for neural network training and testing
'''

from abc import ABCMeta, abstractmethod
import gzip
import numpy as np

## Class that dispenses batches of data for mini-batch training
class BatchDispenser(object):
    ''' BatchDispenser interface cannot be created but gives methods to its
    child classes.'''
    __metaclass__ = ABCMeta

    @abstractmethod
    def read_target_file(self, target_path):
        '''
        read the file containing the targets

        Args:
            target_path: path to the targets file

        Returns:
            A dictionary containing
                - Key: Utterance ID
                - Value: The target sequence as a string
        '''

    def __init__(self, feature_reader, target_coder, size, target_path):
        '''
        Abstract constructor for nonexisting general data sets.

        Args:
            feature_reader: Kaldi ark-file feature reader instance.
            target_coder: a TargetCoder object to encode and decode the target
                sequences
            size: Specifies how many utterances should be contained
                  in each block.
            target_path: path to the file containing the targets
        '''

        #store the feature reader
        self.feature_reader = feature_reader

        #get a dictionary connecting training utterances and targets.
        self.target_dict = self.read_target_file(target_path)

        #detect the maximum length of the target sequences
        self.max_target_length = max([target_coder.encode(targets).size
                                      for targets in self.target_dict.values()])

        #store the batch size
        self.size = size

        #save the target coder
        self.target_coder = target_coder

        self.looped = False

    def have_feature(self):

        '''
        Check if there is still have features

        Returns:
            bool signal:
                True: still have fetures
                False: no feature left
        '''
        if self.looped:
            return False
        else:
            return True

    def init_feature(self):
        '''
            set the sign of looped to true

        '''
        self.looped = False

    def get_feature(self):

        '''
            Get a shuffle block of features and targets
            A pair containing:
                - The features: a shuffle feature matrix
                - the targets: a shuffle target vactor
        '''
        block_inputs, block_targets = self.get_batch()



        inputs = block_inputs[0]
        for i in range(1, len(block_inputs)):
            inputs = np.row_stack((inputs, block_inputs[i]))

        targets = block_targets[0]
        for i in range(1, len(block_targets)):
            targets = np.append(targets, block_targets[i])
        #shuffle the data
        feat_and_target = np.column_stack((inputs, targets))
        np.random.shuffle(feat_and_target)

        inputs = feat_and_target[:,0:inputs.shape[1]]
        targets = feat_and_target[:,inputs.shape[1]:inputs.shape[1]+1]
        return inputs, targets


    def get_batch(self):
        '''
        Get a batch of features and targets.

        Returns:
            A pair containing:
                - The features: a list of feature matrices
                - The targets: a list of target vectors
        '''

        #set up the data lists.
        block_inputs = []
        block_targets = []

        while len(block_inputs) < self.size:
            #read utterance
            utt_id, utt_mat, looped = self.feature_reader.get_utt()
            if looped == True:
                self.looped = looped
            #get transcription
            if utt_id in self.target_dict and utt_mat is not None:
                targets = self.target_dict[utt_id]
                encoded_targets = self.target_coder.encode(targets)

                len_utt = utt_mat.shape[0]
                len_target = encoded_targets.shape[0]
                #check the length of target and frame number

                if abs(len_utt - len_target) <=5:
                    min_len = min(len_utt, len_target)
                    block_inputs.append(utt_mat[0:min_len,])
                    block_targets.append(encoded_targets[0:min_len,])
                else:
                    print 'WARNING no targets for %s' % utt_id
            else:
                if utt_id not in self.target_dict:
                    print 'WARNING no targets for %s' % utt_id
                if utt_mat is None:
                    print 'WARNING %s is too short to splice' % utt_id

        return block_inputs, block_targets


    def split(self):
        '''
        split off the part that has allready been read by the batchdispenser

        this can be used to read a validation set and then split it off from
        the rest
        '''
        self.feature_reader.split()

    def skip_batch(self):
        '''skip a batch'''

        skipped = 0
        while skipped < self.size:
            #read nex utterance
            utt_id = self.feature_reader.next_id()

            if utt_id in self.target_dict:
                #update number skipped utterances
                skipped += 1

    def return_batch(self):
        '''Reset to previous batch'''

        skipped = 0

        while skipped < self.size:
            #read previous utterance
            utt_id = self.feature_reader.prev_id()

            if utt_id in self.target_dict:
                #update number skipped utterances
                skipped += 1

    def compute_target_count(self):
        '''
        compute the count of the targets in the data

        Returns:
            a numpy array containing the counts of the targets
        '''

        #create a big vector of stacked encoded targets
        encoded_targets = np.concatenate(
            [self.target_coder.encode(targets)
             for targets in self.target_dict.values()])

        #count the number of occurences of each target
        count = np.bincount(encoded_targets,
                            minlength=self.target_coder.num_labels)

        return count

    @property
    def num_batches(self):
        '''
        The number of batches in the given data.

        The number of batches is not necessarily a whole number
        '''

        return self.num_utt/self.size

    @property
    def num_utt(self):
        '''The number of utterances in the given data'''

        return len(self.target_dict)

    @property
    def num_labels(self):
        '''the number of output labels'''

        return self.target_coder.num_labels

    @property
    def max_input_length(self):
        '''the maximal sequence length of the features'''

        return self.feature_reader.max_input_length

class TextBatchDispenser(BatchDispenser):
    '''a batch dispenser, which uses text targets.'''

    def read_target_file(self, target_path):
        '''
        read the file containing the text sequences

        Args:
            target_path: path to the text file

        Returns:
            A dictionary containing
                - Key: Utterance ID
                - Value: The target sequence as a string
        '''

        target_dict = {}

        with open(target_path, 'r') as fid:
            for line in fid:
                splitline = line.strip().split(' ')
                target_dict[splitline[0]] = ' '.join(splitline[1:])

        return target_dict

class AlignmentBatchDispenser(BatchDispenser):
    '''a batch dispenser, which uses state alignment targets.'''

    def read_target_file(self, target_path):
        '''
        read the file containing the state alignments

        Args:
            target_path: path to the alignment file

        Returns:
            A dictionary containing
                - Key: Utterance ID
                - Value: The state alignments as a space seperated string
        '''

        target_dict = {}

        with gzip.open(target_path, 'rb') as fid:
            for line in fid:
                splitline = line.strip().split(' ')
                target_dict[splitline[0]] = ' '.join(splitline[1:])

        return target_dict
