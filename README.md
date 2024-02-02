# PCLib

PCLib is a python package with a torch-like API for building and training Predictive Coding Networks. 

The package includes a fully-connected layer implementation, as well as a convolutional one. Both are customisable and can be jointly, or separately used for construction neural networks.

The package also includes some predesigned classification models, each with varying levels of customisability. Some models learn via simultaneously pinning inputs and targets to either end of the network, and then optimising the Free Energy. There are also models which learn a feature extractor via unsupervised learning, and then learn an mlp classifier via backpropagation; these networks have the '_us' suffix. 

Both models and layers are largely customiseable via their initialisation arguments. The fully connected networks are customisable in both width and depth via the initialisation arguments, however the convolutional networks are not. It is worth referring to the documentation to get a better understanding of the API.


## Import
pip install pclib

## Example usage

In the examples folder you will find two different classification tasks which demonstrate the usage of this package.