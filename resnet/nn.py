import tensorflow as tf
import tensorflow.contrib.slim as slim

class nn(object):
    def __init__(self):
        return
    @staticmethod
    def enet_arg_scope(weight_decay=2e-4,
                       batch_norm_decay=0.1,
                       batch_norm_epsilon=0.001):
        '''
        The arg scope for enet model. The weight decay is 2e-4 as seen in the paper.
        Batch_norm decay is 0.1 (momentum 0.1) according to official implementation.

        INPUTS:
        - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
        - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
        - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.

        OUTPUTS:
        - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
        '''
        # Set weight_decay for weights in conv2d and separable_conv2d layers.
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_regularizer=slim.l2_regularizer(weight_decay)):
            # Set parameters for batch_norm.
            with slim.arg_scope([slim.batch_norm],
                                decay=batch_norm_decay,
                                epsilon=batch_norm_epsilon) as scope:
                return scope