import configparser
import tensorflow as tf

# config parameters
config = configparser.ConfigParser()
config.read('config.ini')

# define parameters
input_layer_size = int(config.get('CLASSIFIER', 'INPUT_SIZE'))
output_layer_size = int(config.get('CLASSIFIER', 'OUTPUT_SIZE'))


def inference_ANN(x, prob=1.0, train_flag=False):
    dense_layer_1 = tf.layers.dense(inputs=x, units=2048, activation=tf.nn.relu)
    if train_flag:
        dense_layer_1 = tf.nn.dropout(dense_layer_1, prob)
    dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=512, activation=tf.nn.relu)
    if train_flag:
        dense_layer_2 = tf.nn.dropout(dense_layer_2, prob)
    dense_layer_3 = tf.layers.dense(inputs=dense_layer_2, units=128, activation=tf.nn.relu)
    if train_flag:
        dense_layer_3 = tf.nn.dropout(dense_layer_3, prob)
    dense_layer_4 = tf.layers.dense(inputs=dense_layer_3, units=32, activation=tf.nn.relu)
    if train_flag:
        dense_layer_4 = tf.nn.dropout(dense_layer_4, prob)

    y_ = tf.layers.dense(inputs=dense_layer_4, units=output_layer_size)

    if train_flag:
        return y_
    else:
        return tf.nn.softmax(y_)


def inference_CNN(x, prob=1.0, train_flag=False):
    x_image = tf.reshape(x, [-1, input_layer_size, 1])  # tensor : [N(batch #), W(width), C(Channel)]

    conv_layer_1 = tf.layers.conv1d(inputs=x_image, filters=2, kernel_size=3, padding="same",
                                    activation=tf.nn.relu, data_format="channels_last")
    pool_layer_1 = tf.layers.max_pooling1d(inputs=conv_layer_1, pool_size=2, padding="valid", strides=2)
    # if train_flag:
    #     pool_layer_1 = tf.nn.dropout(pool_layer_1, keep_prob=prob)
    conv_layer_2 = tf.layers.conv1d(inputs=pool_layer_1, filters=4, kernel_size=3, padding="same",
                                    activation=tf.nn.relu, data_format="channels_last")
    pool_layer_2 = tf.layers.max_pooling1d(inputs=conv_layer_2, pool_size=2, padding="valid", strides=2)
    # if train_flag:
    #     pool_layer_2 = tf.nn.dropout(pool_layer_2, keep_prob=prob)
    conv_layer_3 = tf.layers.conv1d(inputs=pool_layer_2, filters=8, kernel_size=3, padding="same",
                                    activation=tf.nn.relu, data_format="channels_last")
    pool_layer_3 = tf.layers.max_pooling1d(inputs=conv_layer_3, pool_size=2, padding="valid", strides=2)
    # if train_flag:
    #     pool_layer_3 = tf.nn.dropout(pool_layer_3, keep_prob=prob)

    convert_flat = tf.layers.flatten(pool_layer_3)

    dense_layer_1 = tf.layers.dense(inputs=convert_flat, units=2048, activation=tf.nn.relu)
    if train_flag:
        dense_layer_1 = tf.nn.dropout(dense_layer_1, prob)
    dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=512, activation=tf.nn.relu)
    if train_flag:
        dense_layer_2 = tf.nn.dropout(dense_layer_2, prob)
    dense_layer_3 = tf.layers.dense(inputs=dense_layer_2, units=128, activation=tf.nn.relu)
    if train_flag:
        dense_layer_3 = tf.nn.dropout(dense_layer_3, prob)
    dense_layer_4 = tf.layers.dense(inputs=dense_layer_3, units=32, activation=tf.nn.relu)
    if train_flag:
        dense_layer_4 = tf.nn.dropout(dense_layer_4, prob)

    y_ = tf.layers.dense(inputs=dense_layer_4, units=output_layer_size)

    if train_flag:
        return y_
    else:
        return tf.nn.softmax(y_)