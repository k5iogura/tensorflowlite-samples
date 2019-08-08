import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Graph().as_default() as graph: # Set default graph as graph

   with tf.Session() as sess:
        # Load the graph in graph_def
        print("load graph")

        # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
        with gfile.FastGFile("mnist.pb",'rb') as f:

            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)

            tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
            )

            # Print the name of operations in the session
            for op in graph.get_operations():
                    print("Operation Name :",op.name)        # Operation name
                    print("Tensor Stats :",str(op.values()))     # Tensor name

            input = tf.placeholder(np.float32, shape = [None, 32, 32, 3], name='input')
            dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')

            tf.import_graph_def(graph_def, {'input': input, 'dropout_rate': dropout_rate})

            output_tensor = graph.get_tensor_by_name("import/cnn/output:0")

