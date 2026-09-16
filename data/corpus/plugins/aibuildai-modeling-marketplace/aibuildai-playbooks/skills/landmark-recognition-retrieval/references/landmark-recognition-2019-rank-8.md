# 8th place solution

Competition: landmark-recognition-2019
Rank: #8
Source: https://www.kaggle.com/c/landmark-recognition-2019/discussion/94512#latest-549367

First of all, congratulations to everyone that participated! This was a rough competition, more than 4 million images (&gt; 600GB) and 200 thousand classes!
And, of course, congratulations to the winners! Well deserved!

# Summary
* **Training set**: Sub-set of 2018 Google Landmark Recognition
* **Descriptor**: Ensemble of DenseNet121-GeM-512FC (512x512, ArcFace) and ResNet101-GeM-2048FC (768x~, ContrastiveLoss)
* **Landmark classifier**: ResNet50 with T-scalling trained on OpenImages and Places. Remove images with less than 0.99 probability from train-set (4M-&gt;2.6M)
* **Inference**: Extract features for train and test set and compute the cosine similarity matrix between test and class centers. Then, apply K-NN to retrieve the 12 nearest neighbours and apply softmax on them.
* **Re-rank 1**: Multiply confidence score from inference by classification's score on test.
* **Re-rank 2**: Perform K-NN search (K=1750) of test set on itself and apply softmax. Multiply last score by this one.

# Long version
I believe two issues must be addressed to tackle this competition. The first is a similarity search problem, where we want to create a good descriptor that would allow us to correctly classify an image based on a K-NN search in a gallery (trainset).
The second part is a detection/re-rank of non-landmarks in both train (gallery) and test (query) sets. This is paramount since the GAP metric used in this competition penalize us if we make a prediction for a non-landmark with higher confidence score than any other landmark images.

## Similarity search
### Descriptor
I reckon a metric learning approach is much more natural to this task than a pure classification one. This is because we have a huge number of classes (&gt; 200k) and some of them have just a few examples. Furthermore, in practice we would not be able to re-train our classifier every time we need to add a new landmark to the gallery.

With this mindeset, I spent most of the time (until the last 2 weeks) training several models with ArcFace loss. I tried several types of pooling schemes such as SPoC, GeM and REMAP (my own re-implementation). Also tried several types of backbones like DenseNet121, ResNet101 and SEResNeXt101.

The ones that worked best for me were DenseNet121-GeM-512FC (512x512) and SEResNeXt101-REMAP-2048FC (384x384). However, SEResNeXt101-REMAP was really slow so I only used DenseNet for the second stage.

I trained on a sub-set of last years' competition, it was around 80k images of 15k classes. A margin of 0.8 for [ArcFace](https://arxiv.org/abs/1801.07698) loss was used and images were resized to the specified size. Weird as it may sound, I found it was better to squash it into a square than keeping the original aspect ratio with zero padding.

I (unfortunately) spent most of my time here and the result were not so great. My best model only score around 0.33 on the image **retrieval** public LB (no QE, DBA or any re-rank). At this time I decided to give up ArcFace and go for the siamese approach.

In the last weeks I trained a ResNet101-GeM-2048FC (768x~) following Radenovic's [paper](https://arxiv.org/abs/1711.02512) and despite I didn't submit to the retrieval competition (deadline was near) I believe it was considerably better than the other models I had trained so far.

My final descriptor was an 2560-dim. ensemble of both models. I L2-normalized both and concatenated as follows: [1.5 * ResNet, 1 * DenseNet]. I didn't have enough time nor RAM to apply PCA on the resultant descriptor.

### Landmark classifier
Since there were a lot of non-landmark images both in test and train sets I trained a landmark classifier using the descriptor's train set as landmark examples. For the non-landmarks, I used images from OpenImages and Places datasets. I used classes such as: sky, forest, bathroom, bedroom, restaurant, Human faces, food, etc...
It was important to use outdoor classes as negative examples, otherwise the model would be biased to predict any outdoor image as landmark. Finally, I used [T-scalling](https://arxiv.org/pdf/1706.04599.pdf) to scale the predictions.

For the train set I removed all the images with classification score less than 0.99, which resulted in around 2.6M images.

For test set, on the other hand, I did not remove any images but used the scores to re-rank the final submission's confidence scores. I did this because, if the conf. score is well ranked, there is no penalty for making predictions on non-landmark images.

### Out-of-distribution images
Since the descriptor was only training on landmark images it is very bad in differentiating between images that are non-landmarks, e.g. human faces or types of food. As consequence of this fact, non-landmark images are matched with very high confidence scores, decreasing the GAP metric. In order to overcome this issue, three re-rank steps were used (will be explained later).

### Extract features
The CNNs previously mentioned are used to extract features (descriptors) for both train and test sets. Then for the train set, the class centers are computed by averaging all the image descriptors that belong to the same class. Finally, both test and class centers descriptors are L2 normalized.

### Search
The K-NN search was performed on the cosine distance matrix using [Faiss](https://github.com/facebookresearch/faiss) library. The top 12 NN were kept and softmax was applied on them.

Even though we only need the first nearest neighbour, it was important to keep more of them to apply softmax and convert the cos. similarities into a probability distribution. By doing this we filter out some non-landmarks that have high similarity with a lot of classes in the gallery (train set). Without this or any other re-ranking steps, the score is likely to be close to zero even for a good descriptor.

### Re-rank:
The second re-rank was very straightforward. The original confidence score was multiplied by the probability of being a landmark from the classifier. With this step many non-landmark images got their conf. score reduced.

### Re-rank based on out-of-distribution similarity
For this step, the K-NN (K=1750) of the test set with itself was computed. The intuition here is that since our model is known to produce high similarity between out-of-distribution images, if we compute the similarity between the test set it is likely that non-landmark images will have much more similar neighbours than landmark ones. Thus, I apply the softmax and multiply by the confidence score from the last step.

Hence, the final score is given by: `score = metric_learning_score * classifier_score * ood_score`

# Leaderboard progression (stage 2)
* **DenseNet121-GeM-512FC (cls re-rank)**: 0.22379
* **DenseNet121-GeM-512FC (cls and ood re-rank)**: 0.23862
* **ResNet101-GeM-2048FC (cls and ood re-rank**): 0.26994
* **Ensemble (cls and ood re-rank)**: 0.28792
# Resources
* **GPU**: 1x Nvidia 1080ti (11GB)
* **RAM**: 32GB
* **HD**: 1TB SSD
* **Processor**: Intel i7 6800k

#Acknowledgments
Thanks Kaggle and Google for hosting this competition. Thanks as well to all the community and @filipradenovic  for opening his 2018 Google-Landmark-Retrieval solution.
