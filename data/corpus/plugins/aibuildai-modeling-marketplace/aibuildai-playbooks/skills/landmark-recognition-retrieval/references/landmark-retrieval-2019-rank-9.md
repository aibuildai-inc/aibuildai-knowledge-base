# 9th Place Solution: VRG Prague

Competition: landmark-retrieval-2019
Rank: #9
Source: https://www.kaggle.com/c/landmark-retrieval-2019/discussion/94843#latest-552745

# Approach by Visual Recognition Group (VRG) Prague

Thanks to the organizers for the great challenge once again and congratulations for the winning teams!

Similarly to last year, our method is based on a combination of CNN-based global descriptors trained on the training set of the first version [Google Landmarks Dataset](https://www.kaggle.com/google/google-landmarks-dataset) used in the challenges last year. Unfortunately, I did not have enough time to dedicate to the challenge, so I did not manage to set up a successful training on the training set of this years [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark).

### Note

Performance is presented in the format private / public score.

### Representation learning

Our network consists of the convolutional layers of ResNet50, ResNet101, and ResNet152 pre-trained on ImageNet, followed by generalized-mean pooling ([GeM](https://arxiv.org/abs/1711.02512)), l2 normalization, a fully-connected (FC) layer, and a final l2 normalization. We train a network with contrastive or triplet loss (when training for a longer time, triplet loss resulted in a superior performance) on pairs selected from the [Google Landmarks Dataset](https://www.kaggle.com/google/google-landmarks-dataset) training set with images resized to have the largest dimensions at most equal to 1024 (900 for ResNet152 due to memory restrictions). We randomly sample at most 50 positive pairs per landmark, while the hard negatives are re-mined for each epoch of the training. The FC layer is initialized by the parameters of [supervised whitening](https://arxiv.org/abs/1604.02426) learned on such pairs. 

Training was done using our publicly available [CNN Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) toolbox. 

Validation was done on [Revisited Oxford and Paris datasets](https://github.com/filipradenovic/revisitop).

During testing, we perform descriptor extraction at 3 scales (scaling factors of ```1, 1/sqrt(2), sqrt(2)```), where the original scale corresponds to largest image size equal to 800 (the maximum image size of v2 dataset), sum-aggregate the descriptors, and l2 normalize. The [PCA whitening with shrinkage](https://arxiv.org/pdf/1811.11147.pdf) is learned/applied on the multi-scale descriptors. The dimensionality of the final descriptor is 2048.

### Simple NN-search

When using our trained networks with cosine similarity and nearest neighbor search the performance is:
```
ResNet101-GeM: 0.200 / 0.178
ResNet152-GeM: 0.196 / 0.167
ResNet50-GeM: 0.184 / 0.163
```

Additionally, we evaluated [ResNet101 with RMAC pre-trained](https://arxiv.org/pdf/1610.07940.pdf):
```
ResNet101-RMAC: 0.181 / 0.150
```

Concatenating these four representations and jointly PCA-reducing (with shrinkage) to 4096 dimensions results in:
```
concat[4096]: 0.213 / 0.186
```

### Iterative DBA and QE

We perform iterative DBA with an increasing number of images used in each iteration. More precisely, we use three iterations, starting with one nearest-neighbor image being used to average with the respective DB image, repeating nearest-neighbor search and adding one additional nearest neighbor in each step. Instead of a plain average, we use weighted average, where the weight is given as squared cosine similarity of the respective DB image and its neighbor. We denote this method as `2dba1:1:3`.

We perform iterative QE in a similar way as DBA, but now in 10 itearation, denoting it as `2qe1:1:10`.

Performance of our concatenated 4096-dimensional representation is now:
```
concat[4096] -&gt; 2dba1:1:3 -&gt; 2qe1:1:10: 0.254 / 0.229
```

### Graph-based QE and Diffusion

We tested two approaches for the final rank computation, graph traversal and diffusion.

Our graph traversal algorithm explores the knn graph with a greedy algorithm where the next image to be retrieved is the one with the highest similarity to any already retrieved image or to the query. This approach is a special case of [Explore-Exploit Graph Traversal for Image Retrieval](http://www.cs.toronto.edu/~mvolkovs/cvpr2019EGT.pdf) where only one image is always exploited. This yielded the score of `0.257 / 0.235`.

For diffusion, we used the conjugate gradient method from the [Python implementation](https://github.com/ducha-aiki/manifold-diffusion) of [Efficient Diffusion on Region Manifolds](https://arxiv.org/pdf/1611.05113.pdf) which was patched to support sparse matrices on the input. This composes our final submission of `0.263 / 0.243` on private / public score.

### [14. 06. 2019] UPDATE

I released our models as a part of our [CNN Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch#networks-with-whitening-learned-end-to-end) toolbox, with the script to evaluate them on R-Oxford and R-Paris.

Multi-scale performance in the following table:

| Model | ROxf (M) | RPar (M) | ROxf (H) | RPar (H) |
|:------|:------:|:------:|:------:|:------:|
| [gl18-tl-resnet50-gem-w](http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth)  | 63.6 | 78.0 | 40.9 | 57.5 |
| [gl18-tl-resnet101-gem-w](http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth) | 67.3 | 80.6 | 44.3 | 61.5 |
| [gl18-tl-resnet152-gem-w](http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth) | 68.7 | 79.7 | 44.2 | 60.3 |
