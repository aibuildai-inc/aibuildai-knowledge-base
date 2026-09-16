# 13th Place Solution for the Google - Fast or Slow? Predict AI Model Runtime Competition

Competition: predict-ai-model-runtime
Rank: #13
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/458370

# 13th Place Solution for the Google - Fast or Slow? Predict AI Model Runtime Competition

This is my solution for the "Fast or Slow? Predict AI Model Runtime" Competition. I hope you find it useful!

The key principles of my approach were to start with something simple and improve it in many iterations, and to work under the hardware constraints that I had (16 GB RAM, 4 GB VRAM).


## Context section

* Business context: https://www.kaggle.com/competitions/predict-ai-model-runtime/overview
* Data Context: https://www.kaggle.com/competitions/predict-ai-model-runtime/data

## Overview of the approach

### Layout model

The core idea is to extract features from the graph and its nodes, and train a Multi-layer Perceptron with that information. For each configurable node of the computational graph we took some node properties, the layout used, and properties from its "parents" and "siblings". The "parent" nodes are the ones that produce the node inputs. Two nodes are "siblings" if they share a parent. For example, in the following figure, nodes 1 and 2 are the parents and nodes 3, 5, and 6 are the siblings.



For each configurable node in the graph we had a list of features that were processed by 3 fully-connected layers with dropout layers in between. After that, the "node" dimension was averaged, so all the node information is represented as a vector. Two extra inputs were concatenated to this vector: a "graph description" and the "subset information". The "graph description" is a vector with how many nodes of each type are present in the graph (normalized to sum 1), with an extra value that is the number of nodes of the graph (with a logarithm) to give the model an idea of the graph's size. The "subset information" is a vector that signals if the graph comes from the "xla:default", "xla:random", "nlp:default" or "nlp:random" subset, for what we used an "Embedding" layer of keras.

This new vector was processed by 3 additional fully-connected layers (no dropout this time). The Pairwise Hinge loss was used as objective function during training. In order to do that, each training batch of size 128 had examples of 16 different graphs, with 8 configuration examples each. Each batch is built by randomly choosing a subset with equal probability, so the batches are subset-balanced on average. The final submission was an ensemble of 3 independent training runs. 

### Tile model

The tile model is a simplified version of the layout model. Each configuration is described with a vector that has the "config_feat" information and also a "graph descriptor" (the same one from the layout model). This model is a Multi-layer Perceptron with 3 fully-connected layers and one dropout layer after the first layer. The loss function and batch structure is the same as the layout model, but with a larger batch size (600) and a larger number of configurations per graph (20).

### Validation

The "valid" folder examples from the dataset were used as the validation set. We performed validation every 10000 training iterations, and we computed the competition metric over that set. In the case of the layout model, the metric was computed for each one of the four subsets and then an average is computed. If the validation metric did not improve after 5 validations, the training is stopped.

## Details of the submission

### Tile model details

The "tile problem" was relatively easy in comparison to the layout problem, so we did not spend that much effort improving this model. The number of configurations per graph was limited to 160, and we sampled them giving the configurations with a lower runtime a higher probability (using an exponential distribution), as the challenge here was about finding the fastest configuration, not sorting all the configurations.

### Node features (Layout model)

From all the available features in the "node_feat" matrices, we selected the ones that we thought were the most important. This helps keeping the required memory low. These features are:

* shape_dimensions (21-26).
* reshape/broadcast dimensions (31-36).
* convolution_dim_numbers_input_spatial_dims (95-98).
* convolution_dim_numbers_kernel_spatial_dims (101-104).
* layout_minor_to_major (134-139)

Values in parentheses correspond to the selected indices of "node_feat". We also gave the node layout information and the opcode to the model (encoded as a vector with a keras Embedding layer). Another important thing is that a re-ordered version of the shapes was given to the network according to the layout information (in addition to the original version). 

For the siblings we took each sibling output shape, its layout and a boolean that compares the node layout with the sibling layout to check if they are the same. As the competition overview mentioned, if the layouts of two siblings are different, an extra copy operation is needed, so that motivated the creation of this variable.

For the parents we keep their output shapes and physical layouts. We also took the opcodes from the parents and the siblings, and express them as a vector with the help of the Embedding layer of keras.

### Keep the training stable (Layout model)

To facilitate the training process, all the features that took values across many orders of magnitude (e.g. tensor shapes) were passed to a logarithm to avoid very large input values. The features were scaled using a mean / standard deviation normalization, with some clipping in the std estimation to avoid dividing by a very small value.

A cosine decay schedule was used for the learning rate. After a 10000 iteration linear warm-up in the learning rate, the cosine decay reduced the parameter across 250k iterations until it reached a 5 % of the original value. Adam was used as optimizer, with a clipnorm value of 1.0 to avoid large weight updates.

Many configurations had the same layout. All of them were replaced by just one instance of that layout, and the runtime replaced by the mean of the runtimes.

### Things that didn't work that well...

We had some instability problems with the List MLE loss, so we chose using the Pairwise hinge loss instead. The problem was that after many iterations, suddenly a NaN value appeared in the model (or loss, idk) and destroyed all the model weights.

### Keeping the training under the memory budget

As we mentioned, we worked with a limited memory budget of 16 GB of RAM and 4 GB of VRAM, so we had to be very careful of not loading too much data at the same time and be conservative with the model size. The first important thing was to process all the npz files and save the necessary information in the tfrecords format of tensorflow (with file compression activated). During training, these files were read from disk, trying to give the model samples from many different graphs instead of seeing just one graph at a time.

The number of configuration per graph was capped at 7500, and the number of configurable nodes given to the network was capped at 1000. As many different graphs were given to the network and considering that each graph has a different number of configurable nodes, it is necessary to pad and mask tensors to make all the samples the same length across the "node" dimension. This can have a heavy memory burden if the number of used nodes is increased too much, so the number 1000 was chosen given this restriction. I also put a limit in the number of parents (2) and siblings (3) for each node.

## Sources

Code: https://github.com/ignacioreyes/kaggle_model_runtime

Embedding layer (tf/keras): https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

#### Note: I wrote many sections in plural, as is customary in academic papers.
