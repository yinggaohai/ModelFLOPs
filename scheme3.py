#coding=utf-8
import numpy as np
from DenseNet import buildingModel
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

##-------------------------------------- config parameter----------------------------------------
# image resize
newShape=(60,60)

# Hyperparameter
epsilon = 1e-8 # AdamOptimizer epsilon

# Label & batch_size
channel_num=1
class_num = 7

## ------------------------------------construct DenseNet-------------------------------------------------
x_image = tf.placeholder(tf.float32, shape=[1, newShape[0],newShape[1],channel_num],name='inputs')
label = tf.placeholder(tf.float32, shape=[1, class_num],name='target')
training_flag = tf.placeholder(tf.bool,name='training_flag')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = buildingModel(batch_images=x_image, class_num=class_num, training_flag=training_flag)
p_fromModel=tf.nn.softmax(logits,name='softmax')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

##yinggh
if True:
    variables=tf.trainable_variables()
    from functools import reduce
    num_variable=[reduce(lambda tx,ty:tx*ty,var.shape) for var in variables]
    num_var=sum(num_variable)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)

##coorperate with BN,20190226,yinggaohai
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops): #保证train_op在update_ops执行之后再执行。
   train = optimizer.minimize(cost)

correct_prediction = tf.equal(tf.argmax(p_fromModel, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##----------------------------------------training Model----------------------------------------------
saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    modelName='model.pb'
    graph = convert_variables_to_constants(sess, sess.graph_def, ["softmax"])
    tf.train.write_graph(graph, 'model', modelName, as_text=False)
