#coding=utf8
'''@file trainer.py
neural network trainer environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from classifiers import seq_convertors
import time
import h5py

class Trainer(object):
    '''General class for the training environment for a neural net graph'''
    __metaclass__ = ABCMeta

    def __init__(self, classifier, input_dim, max_input_length,
                 max_target_length, init_learning_rate, l1_penalty,
                 l2_penalty, momentum, minibatch_size, clip_grad):
        '''
        NnetTrainer constructor, creates the training graph

        Args:
            classifier: the neural net classifier that will be trained
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximal length of the input sequences
            max_target_length: the maximal length of the target sequences
            init_learning_rate: the initial learning rate
            l1_penalty: the penalty param for l1 regularization
            l2_penalty: the penalty param for l2 regularization
            momentum: 
            minibatch_size: determines how many frames are
                processed at a time to limit memory usage
        '''

        self.minibatch_size = minibatch_size
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.momentum = momentum

        #create the graph
        self.graph = tf.Graph()

        #define the placeholders in the graph
        with self.graph.as_default():

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32, shape=[minibatch_size, input_dim],
                name='inputs')
            inputs = self.inputs
            #reference labels
            self.targets = tf.placeholder(
                tf.int32, shape=[minibatch_size,1],
                name='targets')
            targets = self.targets
            #the length of the input sequences
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[minibatch_size],
                name='input_seq_length')

            #the length of all the output sequences
            self.target_seq_length = tf.placeholder(
                tf.int32, shape=[minibatch_size],
                name='output_seq_length')
            #compute the training outputs of the nnetgraph
            trainlogits, logit_seq_length, self.modelsaver, self.control_ops = (
                classifier(inputs, self.input_seq_length, is_training=True
                           , reuse=False, scope='Classifier'))

            #compute the validation output of the nnetgraph
            logits, _, _, _ = classifier(inputs, self.target_seq_length,
                                         is_training=False, reuse=True,
                                         scope='Classifier')

            #get a list of trainable variables in the decoder graph
            params = tf.trainable_variables()
            

            #add the variables and operations to the graph that are used for
            #training


            #the total loss of the entire block
            block_loss = tf.get_variable(
                'block_loss', [], dtype=tf.float32,
                initializer=tf.constant_initializer(0), trainable=False)
    
            #the total frame acc of the entire block
            block_acc = tf.get_variable(
                'block_acc', [], dtype=tf.float32,
                initializer=tf.constant_initializer(0), trainable=False)

            with tf.variable_scope('train_variables'):

                #a variable to scale the learning rate (used to reduce the
                #learning rate in case validation performance drops)
                learning_rate_fact = tf.get_variable(
                    'learning_rate_fact', [],
                    initializer=tf.constant_initializer(1.0), trainable=False)

                #compute the learning rate with exponential decay and scale with the learning rate factor
                learning_rate = tf.train.exponential_decay(
                    init_learning_rate, 0, 1,
                    1.0) * learning_rate_fact
               
                #create the optimizer
                optimizer = tf.train.AdadeltaOptimizer(learning_rate)

            #for every parameter create a variable that holds its gradients
            with tf.variable_scope('gradients'):
                grads = [tf.get_variable(
                    param.op.name, param.get_shape().as_list(),
                    initializer=tf.constant_initializer(0),
                    trainable=False) for param in params]

            with tf.name_scope('train'):
                #the total number of frames that are used in the block
                num_frames = tf.get_variable(
                    name='num_frames', shape=[], dtype=tf.int32,
                    initializer=tf.constant_initializer(0), trainable=False)
                
                #operation to update num_frames
                #pylint: disable=E1101
                update_num_frames = num_frames.assign_add(self.minibatch_size)
                #compute the training loss
                loss, acc = self.compute_loss(targets, trainlogits)

                #operation to half the learning rate
                self.halve_learningrate_op = learning_rate_fact.assign(
                    learning_rate_fact/2).op

                #create an operation to initialise the gradients
                self.init_grads = tf.variables_initializer(grads)

                #the operation to initialise the block loss
                self.init_loss = block_loss.initializer #pylint: disable=E1101
                
                #the operation to initialise the block acc
                self.init_acc = block_acc.initializer

                #the operation to initialize the num_frames
                #pylint: disable=E1101
                self.init_num_frames = num_frames.initializer

                #compute the gradients of the batch
                batchgrads = tf.gradients(loss, params)

                #create an operation to update the block loss
                #pylint: disable=E1101
                self.update_loss = block_loss.assign_add(loss)
                #create an operation to update the block acc
                self.update_acc = block_acc.assign_add(acc)

                #create an operation to update the gradients, the block_loss
                #and do all other update ops
                #pylint: disable=E1101
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.update_gradients_op = tf.group(
                    *([grads[p].assign_add(batchgrads[p])
                       for p in range(len(grads)) if batchgrads[p] is not None]
                      + [self.update_loss] + [self.update_acc] + update_ops + [update_num_frames]),
                    name='update_gradients')

                #create an operation to apply the gradients

                #average the gradients
                meangrads = [tf.div(grad, tf.cast(num_frames, tf.float32),
                                    name=grad.op.name) for grad in grads]

                #clip the gradients
                meangrads = [tf.clip_by_value(grad, -clip_grad, clip_grad)
                             for grad in meangrads]
                #apply the gradients
                self.apply_gradients_op = optimizer.apply_gradients(
                    [(meangrads[p], params[p]) for p in range(len(meangrads))])

            with tf.name_scope('valid'):
                #compute the validation loss
                valid_loss, valid_acc = self.compute_loss(targets, logits)

                #operation to update the validation loss
                #pylint: disable=E1101
                self.update_valid_loss= tf.group(*([block_loss.assign_add(valid_loss), 
                                                                block_acc.assign_add(valid_acc), update_num_frames]))


            #operation to compute the average loss in the batch
            self.average_loss = block_loss/tf.cast(num_frames, tf.float32)

            self.average_acc = block_acc/tf.cast(num_frames, tf.float32)

            # add an operation to initialise all the variables in the graph
            self.init_op = tf.global_variables_initializer()

            #saver for the training variables
            self.saver = tf.train.Saver(tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='train_variables'))

            #create the summaries for visualisation
            self.summary = tf.summary.merge(
                [tf.summary.histogram(val.name, val)
                 for val in params+meangrads]
                + [tf.summary.scalar('loss', self.average_loss)])


        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

        #start without visualisation
        self.summarywriter = None

    @abstractmethod
    def compute_loss(self, targets, logits):
        '''
        Compute the loss

        Creates the operation to compute the loss, this is specific to each
        trainer

        Args:
            targets: a list that contains a Bx1 tensor containing the targets
                for eacht time step where B is the batch size
            logits: a list that contains a BxO tensor containing the output
                logits for eacht time step where O is the output dimension
            
            

        Returns:
            a scalar value containing the total loss
        '''

        raise NotImplementedError("Abstract method")

    def initialize(self):
        '''Initialize all the variables in the graph'''

        self.init_op.run() #pylint: disable=E1101

    def start_visualization(self, logdir):
        '''
        open a summarywriter for visualisation and add the graph

        Args:
            logdir: directory where the summaries will be written
        '''

        self.summarywriter = tf.summary.FileWriter(logdir=logdir,
                                                    graph=self.graph)

    # 计算每一个变量梯度的平均值。
    def average_gradients(tower_grads):
        average_grads = []

        # 枚举所有的变量和变量在不同GPU上计算得出的梯度。
        for grad_and_vars in zip(*tower_grads):
            # 计算所有GPU上的梯度平均值。
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            # 将变量和它的平均梯度对应起来。
            average_grads.append(grad_and_var)
        # 返回所有变量的平均梯度，这个将被用于变量的更新。
        return average_grads

    def update(self, train_hd5f_file):
        '''
        update the neural model with a batch or training data

        Args:
            train_hd5f_file: a hd5f for training data

        Returns:
            the loss of this epoch
        '''
        num_blocks = 0
        epoch_loss = 0.0
        epoch_acc = 0.0
        hd5f = h5py.File(train_hd5f_file, 'r')
 
        num_batchs = hd5f['data'].len() / self.minibatch_size
        input_dim = hd5f['data'].shape[1] - 1
        #feed in the batches one by one and accumulate the gradients and loss
        for k in range(num_batchs):

            batch_inputs = hd5f['data'][ k*self.minibatch_size:(k+1)*self.minibatch_size, 0:input_dim]
            batch_targets = hd5f['data'][ k*self.minibatch_size:(k+1)*self.minibatch_size, input_dim:input_dim+1]
            batch_targets = np.reshape(batch_targets, (self.minibatch_size, 1))

            #pylint: disable=E1101
            self.update_gradients_op.run(feed_dict={self.inputs:batch_inputs, self.targets:batch_targets})

            #apply the accumulated gradients to update the model parameters and
            #evaluate the loss
            if self.summarywriter is not None:

                [summary, _] = tf.get_default_session().run(
                    [ self.summary, self.apply_gradients_op])

                #pylint: disable=E1101
                self.summarywriter.add_summary(summary)

            else:
                [_] = tf.get_default_session().run(
                    [self.apply_gradients_op])


            if k % 1250 == 0 and k > 0:

                #get the loss
                loss = self.average_loss.eval()
                acc = self.average_acc.eval()
                num_blocks += 1
                epoch_loss += loss
                epoch_acc += acc
  
                print "the block cross entroy loss is: ", loss, " the block Frame Accuracy is: ", acc
                self.init_loss.run()
                self.init_acc.run()
                self.init_num_frames.run()

        #reinitialize the gradients and the loss
        self.init_grads.run() #pylint: disable=E1101
        #self.init_loss.run()
        #self.init_num_frames.run()

        return epoch_loss/num_blocks

  
    def evaluate(self, dev_hd5f_file, N_GPU):
        '''
        Evaluate the performance of the neural net

        Args:
            dev_hd5f_file: a hd5f file for dev data

            N_GPU: number of gpus
        Returns:
            the loss of the dev data
        '''
        num_blocks = 0
        epoch_loss = 0.0
        epoch_acc = 0.0
        hd5f = h5py.File(dev_hd5f_file, 'r')
 
        num_batchs = hd5f['data'].len() / self.minibatch_size
        input_dim = hd5f['data'].shape[1] - 1

        #feed in the batches one by one and accumulate the gradients and loss
        for k in range(num_batchs):
            batch_inputs = hd5f['data'][k*self.minibatch_size:
                                        (k+1)*self.minibatch_size,
                                        0:input_dim]

            batch_targets = hd5f['data'][k*self.minibatch_size:
                                        (k+1)*self.minibatch_size, input_dim:input_dim+1]
            batch_targets = np.reshape(batch_targets, (self.minibatch_size, 1))

            #pylint: disable=E1101
            tower_loss = []
            tower_acc = []
            for i in range(N_GPU):
                with tf.device('/gpu:%d' % i):
                    Block_loss, Block_acc, _= self.update_valid_loss.run(
                            feed_dict={self.inputs:batch_inputs[i*self.minibatch_size/N_GPU:(i+1)*self.minibatch_size,],
                                    self.targets:batch_targets[i*self.minibatch_size/N_GPU:(i+1)*self.minibatch_size,]})
                    tower_acc.append(Block_acc)
                    tower_loss.append(Block_loss)

            if k % 1250 == 0 and k  > 0:
                #get the loss
                loss = self.average_loss.eval()
                acc = self.average_acc.eval()
                num_blocks += 1
                epoch_loss += loss
                epoch_acc += acc
                print "the block cross entroy loss is: ", loss, " the block Frame Accuracy is: ", acc
                self.init_loss.run()
                self.init_acc.run()
                self.init_num_frames.run()

            tower_loss = []
            tower_acc = []

        return epoch_loss/num_blocks

    def halve_learning_rate(self):
        '''halve the learning rate'''

        self.halve_learningrate_op.run()

    def save_learning_rate(self):

        raise NotImplementedError("Abstract method")


    def save_model(self,  filename):
        '''
        Save the model

        Args:
            filename: path to the model file
        '''
        self.modelsaver.save(tf.get_default_session(), filename)

    def restore_model(self, filename):
        '''
        Load the model

        Args:
            filename: path where the model will be saved
        '''
        self.modelsaver.restore(tf.get_default_session(), filename)

    def save_trainer(self, filedir, filename):
        '''
        Save the training progress (including the model)

        Args:
            filename: path where the model will be saved
        '''

        self.modelsaver.save(tf.get_default_session(), filedir+filename)
        self.saver.save(tf.get_default_session(), filedir+filename + '_trainvars')
        File = filedir+'mlp_best'
        model_file = open(File, 'w')
        model_file.write(filename) 
        model_file.close()

    def restore_trainer(self, filedir):
        '''
        Load the training progress (including the model)

        Args:
            filename: path where the model will be saved
        '''
        File = filedir + 'mlp_best'
        model_file = open(File,'r')
        filename = model_file.readline()
        filename = filename.split('\n')
        filename = filedir + str(filename[0])
        self.modelsaver.restore(tf.get_default_session(), filename)
        self.saver.restore(tf.get_default_session(), filename + '_trainvars')

class CrossEnthropyTrainer(Trainer):
    '''A trainer that minimises the cross-enthropy loss, the output sequences
    must be of the same length as the input sequences'''

    def compute_loss(self, targets, logits):
        '''
        Compute the loss

        Creates the operation to compute the cross-enthropy loss for every input
        frame (if you want to have a different loss function, overwrite this method)

        Args:
            targets: a Bx1 tensor containing the targets for each time step where B is the batch size
            logits: a BxO tensor containing the output logits for each time step where O is the output dimension
             

        Returns:
            a scalar value containing the loss
        '''
        with tf.name_scope('cross_enthropy_loss'):


            targets = tf.reshape(targets,[-1])
            correct_pred = tf.nn.in_top_k(logits, targets, 1)
            #compute the frame acc
            acc = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
            #one hot encode the targets
            #pylint: disable=E1101
            targets = tf.one_hot(targets,int(logits.get_shape()[1]))

            #compute the cross-enthropy loss
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
            return loss, acc



