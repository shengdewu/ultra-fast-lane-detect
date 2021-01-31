import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


class resnet_tf():

    def __init__(self):
        self.MOVING_AVERAGE_DECAY = 0.9997
        self.BN_DECAY = self.MOVING_AVERAGE_DECAY
        self.BN_EPSILON = 0.001
        self.CONV_WEIGHT_DECAY = 0.00004
        self.CONV_WEIGHT_STDDEV = 0.1
        self.FC_WEIGHT_DECAY = 0.00004
        self.FC_WEIGHT_STDDEV = 0.01
        self.RESNET_VARIABLES = 'resnet_variables'
        self.UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
        self.IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.layer5 = None
        return

    def inference(self, x, is_training,
                  num_classes=1000,
                  num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
                  use_bias=False,  # defaults to using batch norm
                  bottleneck=True):
        c = dict()
        c['bottleneck'] = bottleneck
        c['is_training'] = tf.convert_to_tensor(is_training,
                                                dtype='bool',
                                                name='is_training')
        c['ksize'] = 3
        c['stride'] = 1
        c['use_bias'] = use_bias
        c['fc_units_out'] = num_classes
        c['num_blocks'] = num_blocks
        c['stack_stride'] = 2

        with tf.variable_scope('scale1'):
            c['conv_filters_out'] = 64
            c['ksize'] = 7
            c['stride'] = 2
            x = self.conv(x, c)
            x = self.bn(x, c)
            x = tf.nn.relu(x)
            self.layer1 = x

        with tf.variable_scope('scale2'):
            x = self._max_pool(x, ksize=3, stride=2)
            c['num_blocks'] = num_blocks[0]
            c['stack_stride'] = 1
            c['block_filters_internal'] = 64
            x = self.stack(x, c)
            self.layer2 = x

        with tf.variable_scope('scale3'):
            c['num_blocks'] = num_blocks[1]
            c['block_filters_internal'] = 128
            c['stack_stride'] = 2
            assert c['stack_stride'] == 2
            x = self.stack(x, c)
            self.layer3 = x

        with tf.variable_scope('scale4'):
            c['num_blocks'] = num_blocks[2]
            c['block_filters_internal'] = 256
            x = self.stack(x, c)
            self.layer4 = x

        with tf.variable_scope('scale5'):
            c['num_blocks'] = num_blocks[3]
            c['block_filters_internal'] = 512
            x = self.stack(x, c)
            self.layer5 = x

        # post-net
        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

        if num_classes != None:
            with tf.variable_scope('fc'):
                x = self.fc(x, c)

        return x

    # This is what they use for CIFAR-10 and 100.
    # See Section 4.2 in http://arxiv.org/abs/1512.03385
    def inference_small(self, x,
                        is_training,
                        num_blocks=3,  # 6n+2 total weight layers will be used.
                        use_bias=False,  # defaults to using batch norm
                        num_classes=10):
        c = Config()
        c['is_training'] = tf.convert_to_tensor(is_training,
                                                dtype='bool',
                                                name='is_training')
        c['use_bias'] = use_bias
        c['fc_units_out'] = num_classes
        c['num_blocks'] = num_blocks
        c['num_classes'] = num_classes
        self.inference_small_config(x, c)

    def inference_small_config(self, x, c):
        c['bottleneck'] = False
        c['ksize'] = 3
        c['stride'] = 1
        with tf.variable_scope('scale1'):
            c['conv_filters_out'] = 16
            c['block_filters_internal'] = 16
            c['stack_stride'] = 1
            x = self.conv(x, c)
            x = self.bn(x, c)
            x = tf.nn.relu(x)
            x = self.stack(x, c)

        with tf.variable_scope('scale2'):
            c['block_filters_internal'] = 32
            c['stack_stride'] = 2
            x = self.stack(x, c)

        with tf.variable_scope('scale3'):
            c['block_filters_internal'] = 64
            c['stack_stride'] = 2
            x = self.stack(x, c)

        # post-net
        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

        if c['num_classes'] != None:
            with tf.variable_scope('fc'):
                x = self.fc(x, c)

        return x

    def _imagenet_preprocess(self, rgb):
        """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
        red, green, blue = tf.split(3, 3, rgb * 255.0)
        bgr = tf.concat(3, [blue, green, red])
        bgr -= self.IMAGENET_MEAN_BGR
        return bgr

    def loss(self, logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
        tf.scalar_summary('loss', loss_)

        return loss_

    def stack(self, x, c):
        for n in range(c['num_blocks']):
            s = c['stack_stride'] if n == 0 else 1
            c['block_stride'] = s
            with tf.variable_scope('block%d' % (n + 1)):
                x = self.block(x, c)
        return x

    def block(self, x, c):
        filters_in = x.get_shape()[-1]

        # Note: filters_out isn't how many filters are outputed.
        # That is the case when bottleneck=False but when bottleneck is
        # True, filters_internal*4 filters are outputted. filters_internal is how many filters
        # the 3x3 convs output internally.
        m = 4 if c['bottleneck'] else 1
        filters_out = m * c['block_filters_internal']

        shortcut = x  # branch 1

        c['conv_filters_out'] = c['block_filters_internal']

        if c['bottleneck']:
            with tf.variable_scope('a'):
                c['ksize'] = 1
                c['stride'] = c['block_stride']
                x = self.conv(x, c)
                x = self.bn(x, c)
                x = tf.nn.relu(x)

            with tf.variable_scope('b'):
                x = self.conv(x, c)
                x = self.bn(x, c)
                x = tf.nn.relu(x)

            with tf.variable_scope('c'):
                c['conv_filters_out'] = filters_out
                c['ksize'] = 1
                assert c['stride'] == 1
                x = self.conv(x, c)
                x = self.bn(x, c)
        else:
            with tf.variable_scope('A'):
                c['stride'] = c['block_stride']
                ksize = c['ksize']
                c['ksize'] = 3
                assert c['ksize'] == 3
                x = self.conv(x, c)
                x = self.bn(x, c)
                x = tf.nn.relu(x)
                c['ksize'] = ksize

            with tf.variable_scope('B'):
                c['conv_filters_out'] = filters_out
                ksize = c['ksize']
                c['ksize'] = 3
                stride = c['stride']
                c['stride'] = 1
                assert c['ksize'] == 3
                assert c['stride'] == 1
                x = self.conv(x, c)
                x = self.bn(x, c)
                c['ksize'] = ksize
                c['stride'] = stride

        with tf.variable_scope('shortcut'):
            if filters_out != filters_in or c['block_stride'] != 1:
                c['ksize'] = 1
                c['stride'] = c['block_stride']
                c['conv_filters_out'] = filters_out
                shortcut = self.conv(shortcut, c)
                shortcut = self.bn(shortcut, c)

        return tf.nn.relu(x + shortcut)

    def bn(self, x, c):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        if c['use_bias']:
            bias = self._get_variable('bias', params_shape,
                                      initializer=tf.zeros_initializer)
            return x + bias

        axis = list(range(len(x_shape) - 1))

        beta = self._get_variable('beta',
                                  params_shape,
                                  initializer=tf.zeros_initializer)
        gamma = self._get_variable('gamma',
                                   params_shape,
                                   initializer=tf.ones_initializer)

        moving_mean = self._get_variable('moving_mean',
                                         params_shape,
                                         initializer=tf.zeros_initializer,
                                         trainable=False)
        moving_variance = self._get_variable('moving_variance',
                                             params_shape,
                                             initializer=tf.ones_initializer,
                                             trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, self.BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, self.BN_DECAY)
        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            c['is_training'], lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, self.BN_EPSILON)
        # x.set_shape(inputs.get_shape()) ??

        return x

    def fc(self, x, c):
        num_units_in = x.get_shape()[1]
        num_units_out = c['fc_units_out']
        weights_initializer = tf.truncated_normal_initializer(
            stddev=self.FC_WEIGHT_STDDEV)

        weights = self._get_variable('weights',
                                     shape=[num_units_in, num_units_out],
                                     initializer=weights_initializer,
                                     weight_decay=self.FC_WEIGHT_STDDEV)
        biases = self._get_variable('biases',
                                    shape=[num_units_out],
                                    initializer=tf.zeros_initializer)
        x = tf.nn.xw_plus_b(x, weights, biases)
        return x

    def _get_variable(self, name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype='float',
                      trainable=True):
        "A little wrapper around tf.get_variable to do weight decay and add to"
        "resnet collection"
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.VARIABLES, self.RESNET_VARIABLES]
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               collections=collections,
                               trainable=trainable)

    def conv(self, x, c):
        ksize = c['ksize']
        stride = c['stride']
        filters_out = c['conv_filters_out']

        filters_in = x.get_shape()[-1]
        shape = [ksize, ksize, filters_in, filters_out]
        initializer = tf.truncated_normal_initializer(stddev=self.CONV_WEIGHT_STDDEV)
        weights = self._get_variable('weights',
                                     shape=shape,
                                     dtype='float',
                                     initializer=initializer,
                                     weight_decay=self.CONV_WEIGHT_DECAY)
        return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

    def _max_pool(self, x, ksize=3, stride=2):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')
