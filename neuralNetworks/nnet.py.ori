'''@file nnet.py
contains the functionality for a Kaldi style neural network'''

import shutil
import os
import itertools
import numpy as np
import tensorflow as tf
import classifiers.activation
from classifiers.dnn import DNN
from trainer import CrossEnthropyTrainer
from decoder import Decoder

class Nnet(object):
    '''a class for a neural network that can be used together with Kaldi'''

    def __init__(self, conf, input_dim, num_labels):
        '''
        Nnet constructor

        Args:
            conf: nnet configuration
            input_dim: network input dimension
            num_labels: number of target labels
        '''

        #get nnet structure configs
        self.conf = dict(conf.items('nnet'))

        #define location to save neural nets
        self.conf['savedir'] = (conf.get('directories', 'expdir')
                                + '/' + self.conf['name'])

        if not os.path.isdir(self.conf['savedir']):
            os.mkdir(self.conf['savedir'])
        if not os.path.isdir(self.conf['savedir'] + '/training'):
            os.mkdir(self.conf['savedir'] + '/training')

        #compute the input_dimension of the spliced features
        self.input_dim = input_dim

        if self.conf['batch_norm'] == 'True':
            activation = classifiers.activation.Batchnorm(None)
        else:
            activation = None

        #create the activation function
        if self.conf['nonlin'] == 'relu':
            activation = classifiers.activation.TfActivation(activation,
                                                             tf.nn.relu)

        elif self.conf['nonlin'] == 'sigmoid':
            activation = classifiers.activation.TfActivation(activation,
                                                             tf.nn.sigmoid)

        elif self.conf['nonlin'] == 'tanh':
            activation = classifiers.activation.TfActivation(activation,
                                                             tf.nn.tanh)

        elif self.conf['nonlin'] == 'linear':
            activation = classifiers.activation.TfActivation(activation,
                                                             lambda(x): x)
        else:
            raise Exception('unkown nonlinearity')

        if self.conf['l2_norm'] == 'True':
            activation = classifiers.activation.L2Norm(activation)

        if float(self.conf['dropout']) < 1:
            activation = classifiers.activation.Dropout(
                activation, float(self.conf['dropout']))
            
        self.weight_init = self.conf['weight_init']

        #create a DNN
        self.dnn = DNN(
            num_labels, int(self.conf['num_hidden_layers']),
            int(self.conf['num_hidden_units']), activation,
            self.weight_init, int(self.conf['add_layer_period']) > 0)

    def train(self, train_hdf5_file, dev_hdf5_file, train_max_length, dev_max_length):
        '''
        Train the neural network

        Argvs:
            train_hdf5_file : the taining data
            dev_hdf5_file : the dev data
        '''

        #put the DNN in a training environment
        epoch = int(self.conf['epoch'])
        max_epoch = int(self.conf['max_epoch'])
        halve_learning_rate = int(self.conf['halve_learning_rate']) 
        start_halving_impr = float(self.conf['start_halving_impr'])
        end_halving_impr = float(self.conf['end_halving_impr'])
        trainer = CrossEnthropyTrainer(
            self.dnn, self.input_dim, train_max_length,
            train_max_length,
            float(self.conf['initial_learning_rate']),
            float(self.conf['l1_penalty']),
            float(self.conf['l2_penalty']),
            float(self.conf['momentum']), 
            int(self.conf['minibatch_size']),
            float(self.conf['clip_grad']))
        #start the visualization if it is requested
        if self.conf['visualise'] == 'True':
            if os.path.isdir(self.conf['savedir'] + '/logdir'):
                shutil.rmtree(self.conf['savedir'] + '/logdir')

            trainer.start_visualization(self.conf['savedir'] + '/logdir')

        num_gpus = int(self.conf['n_gpus'])

        #start a tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101

        with tf.Session(graph=trainer.graph, config=config):
            #initialise the trainer
            trainer.initialize()

            #load the neural net if the starting epoch is not 0
            if (epoch > 0):
                trainer.restore_trainer(self.conf['savedir'] + '/training/')

            #do a validation step
            
            validation_loss = trainer.evaluate(dev_hdf5_file, num_gpus)
            print '======================================= validation loss at epoch %d is: %f =============================' % (epoch, validation_loss)

            #start the training iteration
            while (epoch < max_epoch):

                #update the model
                loss = trainer.update(train_hdf5_file)

                #print the progress
                print '======================================= training loss at epoch %d is : %f ==============================' %(epoch, loss)

                #validate the model if required

                current_loss = trainer.evaluate(dev_hdf5_file, num_gpus)
                print '======================================= validation loss at epoch %d is: %f ==========================' % (epoch, current_loss)

                epoch += 1

                if halve_learning_rate == 0:

                    if current_loss < validation_loss:

                        if current_loss > (validation_loss - start_halving_impr):

                            halve_learning_rate = 1
                            trainer.halve_learning_rate()
                            print "================ begining to halve learning rate ================"


                        validation_loss = current_loss
                        pre_loss = loss
                        trainer.save_trainer(self.conf['savedir']
                                            + '/training/', 'iter_' + str(epoch) + '_tr'+str(loss)+'_cv'+str(validation_loss))
                    else:
                        print ('the validation loss is worse, returning to '
                               'the previously validated model with halved '
                               'learning rate')

                        trainer.restore_trainer(self.conf['savedir']+ '/training/')
                        trainer.halve_learning_rate()
                        halve_learning_rate = 1
                        print "================ begining to halve learning rate ================"
                        continue
                else:

                    if current_loss < (validation_loss - end_halving_impr):

                        trainer.halve_learning_rate()
                        pre_loss = loss
                        validation_loss = current_loss

                        trainer.save_trainer(self.conf['savedir']
                                            + '/training/', 'iter_' + str(epoch) + '_tr'+str(loss)+'_cv'+str(validation_loss))

                    else:
                        trainer.restore_trainer(self.conf['savedir'] + '/training/')
                        print ('the validation loss is worse, '
                               'terminating training')
                        break

            #save the final model
            trainer.save_model(self.conf['savedir'] + '/final')


    def decode(self, reader, writer):
        '''
        compute pseudo likelihoods the testing set

        Args:
            reader: a feature reader object to read features to decode
            writer: a writer object to write likelihoods
        '''

        #create a decoder
        decoder = Decoder(self.dnn, self.input_dim, reader.max_input_length)

        #read the prior
        prior = np.load(self.conf['savedir'] + '/prior.npy')

        #start tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        with tf.Session(graph=decoder.graph, config=config):

            #load the model
            decoder.restore(self.conf['savedir'] + '/final')

            #feed the utterances one by one to the neural net
            while True:
                utt_id, utt_mat, looped = reader.get_utt()

                if looped:
                    break
                #compute predictions
                output = decoder(utt_mat)
                #get state likelihoods by dividing by the prior
                output = output/prior
                #floor the values to avoid problems with log
                output = np.where(output == 0, np.finfo(float).eps, output)
                #write the pseudo-likelihoods in kaldi feature format
                writer.write_next_utt(utt_id, np.log(output))

        #close the writer
        writer.close()
