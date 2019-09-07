import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from collections import OrderedDict
from pytorch_transformers import BertForSequenceClassification,BertTokenizer,BertConfig
import argparse
import pdb,numpy as np
import cPickle as cp

class QuantizedLayer(nn.Module):
    def __init__(self, layer, n_clusters, init_method='linear', error_checking=False, name="", fast=False):
        """
        - Come up with initial centroid locations for the layer weights
        - Run k means w.r.t layer weights
            call scikit learn
            
        - Replace original weights with indices to mapped weights
            
            Options:
                1. construct new layer and then manually iterate through and replace each weight with centroid index
                    - these indices should not be differentiable (check that PyTorch won't do this automatically)
                    - iterate through centroid locatoins and assign weight to closest centroid
            
        - Stored mapped weights / centroid locations
            - these should be differentiable
            - nn.ParameterList() ? 
        Args:
            layer (nn.Module): Layer to apply quantization to
            n_clusters (int): Number of clusters. log_2(n_clusters) is the size in bits of each 
                                 cluster index.
            init_method (String): Method to initialize the clusters. Currently only linear init,
                                     the method found to work best for quantization in Han et al. (2015),
                                     is implemented.
            error_checking (bool): Flag for verbose K-means and error checking print statements.
        """
        super(QuantizedLayer, self).__init__()
        self.pruned = type(layer) == QuantizedLayer
        self.weightQuantizing = True
        self.weight, self.weight_table = self.quantize_params(layer.weight, n_clusters, init_method, error_checking, name, fast)
        
        if layer.bias is not None: # TODO - add check to make sure 2 ** 8 isn't more clusters than layer.bias.numel()
            bias_nbits = 2**8 if layer.bias.numel() > 2**8 else 2**4
            self.bias, self.bias_table = self.quantize_params(layer.bias, bias_nbits, init_method, error_checking, name, fast)
        else:
            self.bias = None
    
        
    def init_centroid_weights(self, weights, num_clusters, init_method):
        """ computes initial centroid locations 
        for instance, min weight, max weight, and then spaced num_centroid apart
        returns centroid mapped to value
        Args:
            weights (ndarray): Array of the weights in the layer to be compressed.
            num_clusters (int): Number of clusters (see n_clusters in __init__)
            init_method (String): Cluster initialization method (see init_method
                                     in __init__)
        Returns:
            ndarray: Initial centroid values for K-means clustering algorithm.
        """
        init_centroid_values = []
        if init_method == 'linear':
            min_weight, max_weight = np.min(weights).item() * 10, np.max(weights).item() * 10
            spacing = (max_weight - min_weight) / (num_clusters + 1)
            init_centroid_values = np.linspace(min_weight + spacing, max_weight - spacing, num_clusters) / 10
        else:
            raise ValueError('Initialize method {} for centroids is unsupported'.format(init_method))
        return init_centroid_values.reshape(-1, 1) # reshape for KMeans -- expects centroids, features
        
    def quantize_params(self, params, n_clusters, init_method, error_checking=False, name="", fast=False):
        """ Uses k-means quantization to compress the passed in parameters.
        Args: 
            params (torch.Tensor): tensor of the weights to be quantized
            n_clusters (int): Number of clusters (see n_clusters in __init__)
            init_method (String): Cluster initialization method (see init_method in __init__)
            error_checking (bool): Flag for verbose K-means and error checking print statements.
            fast (bool): 
        Returns:
            (nn.Parameter, nn.Embedding)
            * q_params: The quantized layer weights, which correspond to look up indices for the centroid table.
            * param_table: The centroid table for looking up the weights.
        """
        
        dtypes = {
            4 : torch.int8,
            8 : torch.int16,
            16: torch.int32
        }

        orig_shape = params.shape
        flat_params = params.detach().flatten().numpy().reshape((-1, 1))
        
        if fast:
            centroid_idxs = [[0] for _ in range(len(flat_params))]
            centroid_table = torch.tensor(np.array([[0] for _ in range(n_clusters)]), dtype=torch.float32)
        else:
            # initialization method supported in scikitlearn KMeans
            if init_method == 'random' or init_method == 'k-means++':
                kmeans = MiniBatchKMeans(n_clusters, init=init_method, n_init=1, max_iter=100, verbose=error_checking)
            # initialization method not in scikitlearn
            else:
                init_centroid_values = self.init_centroid_weights(flat_params, n_clusters, init_method)
                kmeans = MiniBatchKMeans(n_clusters, init=init_centroid_values, n_init=1, max_iter=100, verbose=error_checking)
            kmeans.fit(flat_params)
            centroid_idxs = kmeans.predict(flat_params)
            centroid_table = torch.tensor(np.array([centroid for centroid in kmeans.cluster_centers_]), dtype=torch.float32)
#         np.save("{}_init_cluster_centroids.npy".format(name), kmeans.cluster_centers_)
#         compareDistributions(flat_params, 
#                              np.array(kmeans.cluster_centers_), 
#                              plot_title="{} Centroid Distributions".format(name),
#                              path="distributions/{}_centroids.png".format(name), 
#                              show_fig=True)
        
        q_params = nn.Parameter(torch.tensor(centroid_idxs, dtype=dtypes[param.bits]).view(orig_shape), requires_grad=False)
        param_table = nn.Embedding.from_pretrained(centroid_table, freeze=False)
        
        if error_checking:
            print("Layer weights: ", params)
            print("Init centroid values: ", init_centroid_values)
            print("Centroid locations after k means", kmeans.cluster_centers_)
            print("Quantized Layer weights (should be idxs): ", q_params)
            print("Centroid table: ", param_table)
            
        self.weightQuantizing = False
        return q_params, param_table
        
    def forward(self, input_):
        """
        - Somehow replace centroid locations in stored matrix with true centroid weights
        - If that doesn't work, construct PyTorch `Function` https://pytorch.org/docs/master/notes/extending.html
        
        Args:
            input_ (torch.Tensor): Input for the forward pass (x value)
        Returns:
            torch.Tensor: Output of the model after run on the input
        """
        orig_weight_shape, orig_bias_shape = self.weight.shape, self.bias.shape
        weights = self.weight_table(self.weight.flatten().long()).view(orig_weight_shape)
        bias = self.bias_table(self.bias.flatten().long()).view(orig_bias_shape) if self.bias is not None else None
        out = F.linear(input_, weights, bias=bias)
        return out

def layer_check(model, numLin):
    """ Checks that there are no linear layers in the quantized model, and checks that the number of 
    quantized layers is equal to the number of initial linear layers.
    
    Args:
        model (nn.Module): Quantized model
        numLin (int): Number of linear layers in the original model
    """
    numQuant = 0
    numBin = 0
    for l in model.modules():
        # if type(l) == nn.Linear and l.name != "classifier":  
        #     raise ValueError('There should not be any linear layers in a quantized model: {}'.format(model))
        if type(l) == QuantizedLayer:
            numQuant += 1
    if (numQuant) != numLin:
        raise ValueError('The number of quantized layers ({}) plus the number of binarized layers ({}) should be equal to the number of linear layers ({})'.format(
            numQuant, numBin, numLin))

def quantize(model, num_centroids, error_checking=False, fast=False):
    """
    1. Iterates through model layers forward
    
    2. For each layer in the model
    
        2.a Replace the layer with a QuantizedLayer
    
    Args:
        model (nn.Module): Model to quantize
        num_centroids (int): See n_clusters in QuantizedLayer().__init__()
    Returns:
        nn.Module: model with all layers quantized
    """
    if error_checking:
        num_linear = len([l for l in model.modules() if type(l) == nn.Linear])
        print("original model: ", model)
        print("number of linear layers in original model: ", num_linear) 
        print("=" * 100)
    
    for name, layer in tqdm(model.named_children()):
        if type(layer) == nn.Linear and name != "classifier":
#             if "linear_" in name:
            # print(name)
            model.__dict__['_modules'][name] = QuantizedLayer(layer, num_centroids, name=name, fast=fast)
        else:
            layer_types = [type(l) for l in layer.modules()]
            if nn.Linear in layer_types:
                quantize(layer, num_centroids, error_checking, fast)
            
    if error_checking:
        layer_check(model, num_linear)
        
    return model

def load_artifacts(model_path):
    """ Loads pretrained model , tokenizer , config."""
    model_class = BertForSequenceClassification
    model = model_class.from_pretrained(model_path)
    
    # model = torch.load(model_path,map_location='cpu')
    # model.to("cpu")
    model.eval()
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Transformer Kmeans quantizer')
    parser.add_argument('-m','--model_folder',action="store", default=False,help="Folder where the model artifacts are saved")
    parser.add_argument('-b','--bits',action="store", default=8,type=int)
    parser.add_argument('-o', action="store",type=str)
    param = parser.parse_args()
    
    model = load_artifacts(param.model_folder)
    quantized_model = quantize(model, 2**param.bits, error_checking=False)
    torch.save(quantized_model,param.o)
    print("=" * 100)
    print(" {} model saved.".format(param.o))