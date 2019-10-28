#coding=utf-8
import tensorflow as tf
def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

# with tf.Graph().as_default() as graph:
#     x=tf.placeholder(shape=[1,10,10,1],dtype=tf.float32)
#     filter = tf.get_variable(initializer=tf.constant_initializer(value=1, dtype=tf.float32), shape=(2,2,1,1), name='A')
#     a=tf.nn.conv2d(x,filter,[1,1,1,1],'SAME')

graph=load_pb('./model/model.pb')
stats_graph(graph)