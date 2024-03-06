# PCLib


PCLib is a python package with a torch-like API for building and training Predictive Coding Networks.<br> 
Documentation can be found [here](https://joeagriffith.github.io/pclib/).

The package includes a fully-connected layer implementation, as well as a convolutional one. Both are customisable and can be used together or separately for building neural networks.

The package also includes a helper class for constructing fully-connected PCNs. This class has been designed to be extremely customiseable such that the network it builds can be used in a wide range of tasks: supervised/unsupervised, classic/inverted, etc. There is also a CNN class, however it is not customisable in shape. For more detailed explanations, please see the documentation.

## Installation
```
pip install pclib
```

## Example usage

In the examples folder you will find two different classification tasks which demonstrate the usage of this package.