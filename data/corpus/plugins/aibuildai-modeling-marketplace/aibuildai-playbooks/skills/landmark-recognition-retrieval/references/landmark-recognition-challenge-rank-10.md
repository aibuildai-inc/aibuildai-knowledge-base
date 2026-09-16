# Our 10th place solution

Competition: landmark-recognition-challenge
Rank: #10
Source: https://www.kaggle.com/c/landmark-recognition-challenge/discussion/58050

Our solution consists of 4 main steps:

**1. CNN**
We experimented with a several types of network architectures: ResNet, Wide ResNet, Inception-v3, DenseNet. We tried to learn both pretrained on ImageNet models and from scratch. The best result on validation was shown by ResNet-101 pretrained on ImageNet. 

As training time augmentation we used random resized crops, color jittering, horizontal flipping. It took us some time to find the parameters for random resizing cause they are important enough. CNNs were trained on 224x224 crops. 

**2. Loss**
We attack landmarks recognition problem from a metric learning point of view. Therefore, the most important part of our algorithm is a suitable for metric learning loss function. After some experiments we choose loss from [ArcFace][1] as the most appropriate for our approach. We trained our networks from previous section with combination of this loss and softmax.

**3. Inference**
For inference we calculated centroid for each landmark class on embeddings from a last layer of CNN by following steps:

 1. Pick 100 random elements of this class from training class
 2. Calculate mean embedding vector on picking elements 
 3. Normalize mean vector 

Additionally on the inference step was used test time augmentation by ten crops averaging. TTA gave us improve about 0.1 GAP.

On inference step the classification decision is made on the basis of the following steps:

 1.  An input image forward through the network
 2. The resultant embedding compares by cosine distance with all classes centroids
 3. Сlass of the nearest centroid by this distance is taken as a result
 4. If not a landmark class softmax value &gt; 0.25 then we consider an input as a not a landmark 

**7. Ensembles**
We trained this pipeline for several types of networks listed in the section 1. Additionally training set was split on 5 folds and each model was trained by cross fold and the result was obtained by averaging across results on this 5 folds. 

For ensembling all these models we made weighted voting across models results. Weights was picked accordingly success of each single model on the LB.

Unfortunately ensemble did not improve our best single model solution by as much as we expected. Our best ResNet-101 model achieve GAP 0.205 (private) when our final result with ensemble is 0.228 (private). 

We also experimented with kNN on embeddings of our best model, but it did not significantly improve the score.


  [1]: https://arxiv.org/abs/1801.07698
