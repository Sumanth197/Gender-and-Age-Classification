import tensorflow as tf


# based on paper: Age and Gender Classification using Convolutional Neural Networks.
def net_paper(images, training_or_not, out_units):

    use_regular = False
    regular_rate = 0.1
    if use_regular:
        l2_regular = tf.contrib.layers.l2_regularizer(regular_rate)
    else:
        l2_regular = None

    def lkrelu(feature):
        alpha = 0.5
        return tf.maximum(alpha * feature, feature)

    activation = tf.nn.selu
    with tf.variable_scope('block1'):
        x = tf.layers.conv2d(images, 96, [7, 7], [4, 4], padding='VALID', activation=activation, name='conv',
                             kernel_regularizer=l2_regular)
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool')
        x = tf.nn.lrn(x, alpha=0.0001, name='lrn')

    with tf.variable_scope('block2'):
        x = tf.layers.conv2d(x, 256, [5, 5], [1, 1], padding='SAME', activation=activation, name='conv',
                             kernel_regularizer=l2_regular)
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool')
        x = tf.nn.lrn(x, alpha=0.0001, name='lrn')

    with tf.variable_scope('block3'):
        x = tf.layers.conv2d(x, 384, [3, 3], [1, 1], padding='SAME', activation=activation, name='conv',
                             kernel_regularizer=l2_regular)
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool')

    with tf.variable_scope('dense_block'):
        x = tf.layers.flatten(x, name='flatten')
        x = tf.layers.dense(x, 512, activation=activation, name='dense1',
                            kernel_regularizer=l2_regular)
        x = tf.layers.dropout(x, rate=0.5, training=training_or_not, name='dropout1')
        x = tf.layers.dense(x, 512, activation=activation, name='dense2',
                            kernel_regularizer=l2_regular)
        x = tf.layers.dropout(x, rate=0.5, training=training_or_not, name='dropout2')

    y = tf.layers.dense(x, out_units, activation=None, name='y')
    return y



model = net_paper





