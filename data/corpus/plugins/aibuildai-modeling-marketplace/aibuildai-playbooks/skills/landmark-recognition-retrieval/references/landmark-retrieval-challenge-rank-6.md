# 6th Place Solution: VRG Prague

Competition: landmark-retrieval-challenge
Rank: #6
Source: https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/58482

# Approach by Visual Recognition Group (VRG) Prague
Thanks to the organizers for the interesting challenge and kudos for the top ranked teams!

### Super quick summary
Our method is based on CNN-based global descriptors trained on the training set of the recognition challenge, while retrieval is performed by diffusion-based similarity search. 

** Performance is presented in the format public / private score.

![pipeline][1]

### GeM descriptor training

Our network consists of the convolutional layers of ResNet101 pre-trained on ImageNet, followed by generalized-mean pooling ([GeM](https://arxiv.org/abs/1711.02512)), l2 normalization, a fully-connected (FC) layer, and a final l2 normalization. We train a two-branch network with contrastive loss on pairs selected from the [Google Landmark Recognition Challenge](https://www.kaggle.com/c/landmark-recognition-challenge) training set with image resized to have the largest dimensions at most equal to 1024. We randomly sample at most 50 positive pairs per landmark, while the hard negatives are re-mined for each epoch of the training. The FC layer is initialized by the parameters of [supervised whitening](https://arxiv.org/abs/1604.02426) learned on such pairs. 

Training was done using our publicly available [CNN Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) toolbox. 

We additionally learn descriptor whitening without supervision on the index set. In particular, we use PCA whitening with shrinkage (work submitted to IJCV, extension of our [BMVC 2017](https://arxiv.org/pdf/1707.07825.pdf)).

### Descriptor extraction
During testing, we perform descriptor extraction at 3 scales (scaling factors of ```1, 1/sqrt(2), sqrt(2)```), sum-aggregate the descriptors, and l2 normalize. The PCA whitening with shrinkage is learned/applied on the multi-scale descriptors. The dimensionality of the final descriptor is 2048.

### Search

We use [diffusion](https://arxiv.org/pdf/1611.05113.pdf), a graph-based query expansion technique, to perform retrieval. We initially construct an affinity matrix with reciprocal k-nearest neighbors (k=50), and then initiate diffusion by the 5 Euclidean nearest neighbors of the query descriptor. 

We use our own [MATLAB implementation](https://github.com/ahmetius/diffusion-retrieval), while a third-party [Python implementation](https://github.com/ducha-aiki/manifold-diffusion) is also available.

** Concatenating our descriptor with [R-MAC trained](https://arxiv.org/pdf/1610.07940.pdf) on an external dataset provides additional improvements. This composes our final submission of 59.1 / 57.6 on public / private score.

### Timing

**Descriptor extraction:** 170 ms per image (GPU).

**Query time:** 12 ms for Euclidean search (CPU) + 150 ms for diffusion per query (GPU).

** Machine used for the search: Intel Xeon CPU E5-2630 v2 @ 2.60GHz with 256 GB of memory (12 threads), with one GPU Tesla P100 with 16 GB of memory.

### Team members
[Filip Radenovic](http://cmp.felk.cvut.cz/~radenfil/), [Ahmet Iscen](http://cmp.felk.cvut.cz/~iscenahm/), [Giorgos Tolias](http://cmp.felk.cvut.cz/~toliageo/), [Ondrej Chum](http://cmp.felk.cvut.cz/~chum/)


  [1]: http://cmp.felk.cvut.cz/cnnimageretrieval/img/google_landmark_retrieval_pipeline.png
