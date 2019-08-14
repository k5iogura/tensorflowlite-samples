from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants

input_saved_model_dir = "./models"
output_node_names = "output"
input_binary = False
input_saver_def_path = False
restore_op_name = None
filename_tensor_name = None
clear_devices = False
input_meta_graph = False
checkpoint_path = None
input_graph_filename = None
saved_model_tags = tag_constants.SERVING

output_graph_filename='mnist.frozen.pb'
freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
    input_binary, checkpoint_path, output_node_names,
    restore_op_name, filename_tensor_name,
    output_graph_filename, clear_devices, "", "", "",
    input_meta_graph, input_saved_model_dir,
    saved_model_tags)
