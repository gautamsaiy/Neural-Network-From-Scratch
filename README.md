# Neural Networks From Scratch

## Goal
1. Learn and understand how machine learning and neural networks work by creating a machine learning framework from scratch (using numpy)
2. Create and train a model that successfully identifies numbers from the MNIST dataset

## Table of Contents
- [Tensor](#tensor)
- [Layers](#layers)
- [Optimizers](#optimizer)
- [Loss](#loss)
- [Dataloaders](#dataloaders)
- [Nets](#nets)


## Tensor
- Inherits from np.ndarray
- "sources" is the data needed to create the tensor
- "grad_fn" is the function that computes the gradient of the tensor
- "requires_grad" specifies if the tensor needs to be differentiated
- "grad" is the gradient of the tensor
- "backward" computes the gradient of the tensor
    - adds the given gradient to its own gradient
    - calls the gradient function
    - recursively calls backward on all its sources


## Layers
- Layer
    - A base class for all layers
    - Has a forward and backward pass
- Dense (Same as Linear)
    - A fully connected layer
    - Performs a matrix multiplication between the input and the weights
- Bias
    - A layer that adds a bias to the input
- ReLU
    - A layer that performs a ReLU activation function


## Optimizers
- Optimizer
    - A base class for all optimizers
    - Has a step function that updates the weights
    - Has a zero_grad function that resets the gradients
- SGD 
    - A stochastic gradient descent optimizer
    - Does a simple gradient * learning rate update to the weights
- Adam
    - An Adam optimizer
    - Does a more complex gradient update to the weights


## Loss
- Functions that compute the loss between the output and the target
- MSE
    - Mean Squared Error
    - Computes the mean squared error between the output and the target
    - $L = \frac{\Sigma(\hat{y} - y)}{2 * |y|}$


## DataLoaders
- DataLoader
    - A base class for all data loaders
- MathFunc
    - A data loader that creates and outputs batches of data from a mathematical function
    - The mathematical function is defined in the constructor


## Nets
- A base class for all neural networks
- Stores layers in a ordered dictionary called "parameters"
- Has a forward pass function that computes the output of the network

