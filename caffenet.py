from kaffe.tensorflow import Network
import tensorflow as tf

class CaffeNetImported(Network):
    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .conv(7, 7, 256, 4096, name='fc6')
             .conv(1, 1, 4096, 4096, name='fc7')
             .conv(1, 1, 4096, 21, relu=False, name='fc8')
             )

def conv2d_depthwise(input, filter, group):
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    input_groups = tf.split(value=input, num_or_size_splits=group, axis=3)
    kernel_groups = tf.split(value=filter, num_or_size_splits=group, axis=3)
    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    # Concatenate the groups
    conv = tf.concat(values=output_groups, axis=3, name='depth_wise_concat')
    return conv

import pdb

def alexfcn(input, n_classes, do_rate, reuse=False):
    #pdb.set_trace()
    input = tf.image.pad_to_bounding_box(input, 100, 100, 424, 424)

    with tf.name_scope('conv1'):
        with tf.variable_scope('conv1', reuse=reuse):
            W_conv1 = tf.get_variable('weights', [11,11,3,96])
            b_conv1 = tf.get_variable('biases', [96])
        conv1 = tf.nn.conv2d(input=input, filter=W_conv1, strides=[1,4,4,1], padding='SAME')
        relu1 = tf.nn.relu(conv1 + b_conv1)

    pool1 = tf.nn.max_pool(value=relu1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')

    lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=2, alpha=0.0001, beta=0.75, bias=.1, name='lrn1')

    with tf.name_scope('conv2'):
        with tf.variable_scope('conv2', reuse=reuse):
            W_conv2 = tf.get_variable('weights', [5,5,48,256])
            b_conv2 = tf.get_variable('biases', [256])
        #conv2 = tf.nn.depthwise_conv2d(input=lrn1, filter=W_conv2, strides=[1,1,1,1], padding='VALID')
        conv2 = conv2d_depthwise(input=lrn1, filter=W_conv2, group=2)
        relu2 = tf.nn.relu(conv2 + b_conv2)

    pool2 = tf.nn.max_pool(value=relu2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')

    lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2, alpha=0.0001, beta=0.75, bias=.1, name='lrn2')

    with tf.name_scope('conv3'):
        with tf.variable_scope('conv3', reuse=reuse):
            W_conv3 = tf.get_variable('weights', [3,3,256,384])
            b_conv3 = tf.get_variable('biases', [384])
        conv3 = tf.nn.conv2d(input=lrn2, filter=W_conv3, strides=[1,1,1,1], padding='SAME')
        relu3 = tf.nn.relu(conv3 + b_conv3)

    with tf.name_scope('conv4'):
        with tf.variable_scope('conv4', reuse=reuse):
            W_conv4 = tf.get_variable('weights', [3,3,192,384])
            b_conv4 = tf.get_variable('biases', [384])
        conv4 = conv2d_depthwise(relu3, W_conv4, 2)
        relu4 = tf.nn.relu(conv4 + b_conv4)

    with tf.name_scope('conv5'):
        with tf.variable_scope('conv5', reuse=reuse):
            W_conv5 = tf.get_variable('weights', [3,3,192,256])
            b_conv5 = tf.get_variable('biases', [256])
        conv5 = conv2d_depthwise(relu4, W_conv5, 2)
        relu5 = tf.nn.relu(conv5 + b_conv5)

    pool3 = tf.nn.max_pool(value=relu5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool3')

    with tf.name_scope('fc6'):
        with tf.variable_scope('fc6', reuse=reuse):
            W_conv6 = tf.get_variable('weights', [7,7,256,4096])
            b_conv6 = tf.get_variable('biases', [4096])
        conv6 = tf.nn.conv2d(input=pool3, filter=W_conv6, strides=[1,1,1,1], padding='SAME')
        relu6 = tf.nn.relu(conv6 + b_conv6)

    #dropout1 = tf.nn.dropout(relu6, do_rate, name='dropout1')

    with tf.name_scope('fc7'):
        with tf.variable_scope('fc7', reuse=reuse):
            W_conv7 = tf.get_variable('weights', [1,1,4096,4096])
            b_conv7 = tf.get_variable('biases', [4096])
        conv7 = tf.nn.conv2d(input=relu6, filter=W_conv7, strides=[1,1,1,1], padding='SAME')
        relu7 = tf.nn.relu(conv7 + b_conv7)

    #dropout2 = tf.nn.dropout(relu7, do_rate, name='dropout2')

    with tf.name_scope('fc8'):
        with tf.variable_scope('fc8', reuse=reuse):
            W_conv8 = tf.get_variable('weights', [1,1,4096,n_classes])
            b_conv8 = tf.get_variable('biases', [n_classes])
        conv8 = tf.nn.conv2d(input=relu7, filter=W_conv8, strides=[1,1,1,1], padding='SAME')
        output = conv8 + b_conv8

    sizes = [input, relu1, pool1, relu2, pool2, relu3, relu4, relu5, pool3, relu6, relu7, output]
    sizes = [tf.shape(layer) for layer in sizes]

    return sizes, input, output

