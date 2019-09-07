import torch.onnx
import torch
from pytorch_transformers import BertForSequenceClassification,BertTokenizer,BertConfig
import pdb,numpy as np
import cPickle as cp

def load_artifacts(model_path):
    """ Loads pretrained model , tokenizer , config."""
    model_class = BertForSequenceClassification
    model = model_class.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    model.to("cpu")
    model.eval()
    return model,tokenizer,config

model,tokenizer,config = load_artifacts("output/")
sample_inputs = cp.load(open("sample_inputs.pkl","r"))

torch.onnx.export(model,tuple(sample_inputs.values()),"BERT_ABSA.onnx")
