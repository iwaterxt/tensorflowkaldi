'''@file feature_reader.py
reading features and applying cmvn and splicing them'''

import ark
import numpy as np
import readfiles

class FeatureReader(object):
    '''Class that can read features from a Kaldi archive and process
    them (cmvn and splicing)'''

    def __init__(self, scpfile, max_input_length):
        '''
        create a FeatureReader object

        Args:
            scpfile: path to the features .scp file
            max_input_length: the maximum length of all the utterances in the
                scp file
        '''

        #create the feature reader
        self.reader = ark.ArkReader(scpfile)

        #store the max length
        self.max_input_length = max_input_length

    def get_utt(self):
        '''
        read the next features from the archive, normalize and splice them

        Returns:
            the normalized and spliced features
        '''
        end_of_scp = False
        #read utterance
        (utt_id, utt_mat, looped) = self.reader.read_next_utt()

        return utt_id, utt_mat, looped

    def next_id(self):
        '''
        only gets the ID of the next utterance

        moves forward in the reader

        Returns:
            the ID of the uterance
        '''

        return self.reader.read_next_scp()

    def prev_id(self):
        '''
        only gets the ID of the previous utterance

        moves backward in the reader

        Returns:
            the ID of the uterance
        '''

        return self.reader.read_previous_scp()

    def split(self):
        '''split of the features that have been read so far'''

        self.reader.split()

def apply_cmvn(utt, stats):
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

    #compute mean
    mean = stats[0, :-1]/stats[0, -1]

    #compute variance
    #variance = stats[1, :-1]/stats[0, -1] - np.square(mean)

    #return mean and variance normalised utterance
    return np.subtract(utt, mean)
    #return np.divide(np.subtract(utt, mean), np.sqrt(variance))

def splice(utt, context_width):
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

    #return None if utterance is too short
    if utt.shape[0]<1+2*context_width:
        return None

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

    return utt_spliced

def apply_global_transform(utt, stats):

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

    #compute mean
    mean = stats[0, :-1]/stats[0, -1]

    #compute variance
    variance = stats[1, :-1]/stats[0, -1] - np.square(mean)

    #return mean and variance normalised utterance
    return np.divide(np.subtract(utt, mean), np.sqrt(variance))
