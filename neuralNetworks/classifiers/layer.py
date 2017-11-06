'''@file layer.py
Neural network layers '''

import tensorflow as tf

class FFLayer(object):
    '''This class defines a fully connected feed forward layer'''

    def __init__(self, output_dim, activation, weight_init, weights_param=None):
        '''
        FFLayer constructor, defines the variables
        Args:
            output_dim: output dimension of the layer
            activation: the activation function
            weights_init: the method to initialize the weights
            weights_param: the standart deviation of the weights by default the
            inverse square root of the input dimension is taken
        '''
        #save the parameters
        self.output_dim = output_dim
        self.activation = activation
        self.weight_init = weight_init
        self.weights_param = weights_param

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):
        '''
        Do the forward computation
        Args:
            inputs: the input to the layer
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the variable scope of the layer
        Returns:
            The output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            with tf.variable_scope('parameters', reuse=reuse):

                if self.weight_init == 'normal':
                    stddev = (self.weights_param if self.weights_param is not None
                                else 1/int(inputs.get_shape()[1])**0.5)
                    weights = tf.get_variable(
                        'weights', [inputs.get_shape()[1], self.output_dim],
                        initializer=tf.random_normal_initializer(stddev=stddev))

                    biases = tf.get_variable(
                        'biases', [self.output_dim],
                        initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.1))

                elif self.weight_init == 'uniform':
                    Range = (self.weights_param if self.weights_param is not None
                                else 1/int(inputs.get_shape()[1])**0.5)
                    weights = tf.get_variable(
                        'weights', [inputs.get_shape()[1], self.output_dim],
                        initializer=tf.random_uniform_initializer(minval=-Range,maxval=Range))

                    biases = tf.get_variable(
                        'biases', [self.output_dim],
                        initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.1))
                else:
                    raise Exception('unkown initialization method')

            #apply weights and biases
            with tf.variable_scope('linear', reuse=reuse):
                linear = tf.matmul(inputs, weights) + biases

            #apply activation function
            with tf.variable_scope('activation', reuse=reuse):
                outputs = self.activation(linear, is_training, reuse)

        return outputs
