import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

with tf.Graph().as_default() as graph: # Set default graph as graph

   with tf.Session() as sess:
        # Load the graph in graph_def
        print("load graph")

        with open('mnist_model_20000_epoch_50_batch.pb','rb') as g:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(g.read())
            _ = tf.import_graph_def(graph_def,name='')

        image_placeholder = tf.get_default_graph().get_tensor_by_name('Placeholder:0')

        inference_op = tf.get_default_graph().get_tensor_by_name('output:0')

        print(image_placeholder)
        #batch = mnist.train.next_batch(50)
        #_ = sess.run(
            #[inference_op],
            #feed_dict={image_placeholder:batch[0]}
        #)

