import onnx
from onnx_tf.backend import prepare
import cPickle as cp
import torch

onnx_model = onnx.load("BERT_ABSA.onnx")
tf_rep = prepare(onnx_model)

# Print out tensors and placeholders in model (helpful during inference in TensorFlow)
print(tf_rep.tensor_dict)

# Export model as .pb file
tf_rep.export_graph('./models/BERT_ABSA_tf.pb')