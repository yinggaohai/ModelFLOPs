#coding=utf-8
import tensorflow as tf
# from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
# from tensorflow.contrib.framework import arg_scope
# import tensorflow.contrib as tf_contrib
# import tflearn
import numpy as np

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    # It is global average pooling without tflearn

    # return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not



def Batch_Normalization(x, training, scope):
    with tf.name_scope(scope):
        x=tf.layers.batch_normalization(x, training=training)
        return x

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'): #'SAME' yinggh
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'): #'SAME' yinggh
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x,class_num) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')


class DenseNet():
    def __init__(self, x, nb_blocks, filters, training,dropout_rate,class_num):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.dropout_rate=dropout_rate
        self.class_num=class_num
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            variables = tf.trainable_variables()    #yinggaohai
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            variables = tf.trainable_variables()    #yinggaohai
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)  #delete?

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training) #delete?

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            dimension_x=x.get_shape().as_list()[-1]
            x = conv_layer(x, filter=int(dimension_x * 0.5), kernel=[1,1], layer_name=scope+'_conv1') #fileter num should be a half of X channel
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2,padding='VALID')

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3,3], stride=2,padding='VALID')

        # for i in range(self.nb_blocks) :
        #     # 6 -> 12 -> 48
        #     x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
        #     x = self.transition_layer(x, scope='trans_'+str(i))



        x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        # x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        # x = self.transition_layer(x, scope='trans_3')


        x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_final')

        # 100 Layer
        # x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x,self.class_num)

        return x

def buildingModel(batch_images=None, class_num=None ,training_flag=None):
    nb_block = 3
    growth_k = 12
    dropout_rate = 0.2  # 0.3 is a good value
    logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag,
                      dropout_rate=dropout_rate,class_num=class_num).model
    print('buildingModel completed.')
    return logits
