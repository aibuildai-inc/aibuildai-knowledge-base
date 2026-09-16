# 1st Place Solution

Competition: landmark-recognition-2020
Rank: #1
Source: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187821

Thanks to google and kaggle for hosting this yearly competition. It was a lot of fun exploring and learning all about global, local descriptors and algorithms to match similar images.

### Brief Summary

Our solution is an ensemble of 7 global descriptor only models trained with arcface loss. We classify landmarks by KNN on an extended version of the train dataset and efficiently rerank predictions and filter noise using cosine similarity to non-landmark images. We did not use any local descriptors.

### Detailed Summary

Below we give a detailed description of our solution of which architecture is only a small part.
Video content of us presenting the solution is available under:

NVIDIA Grandmaster Series Ep2 https://youtu.be/VxNDH6qLZ_Q
Chai Time Data Science https://youtu.be/NRl3lMlixPc


#### Pipeline
We wanted to use this competition as a chance to improve our pipeline and coding skills. While in past competitions we mainly used jupyter notebooks locally, we switched to a collaborative approach using scripts with github versioning for this one. After some acclimatization we clearly saw a benefit of our pipeline consisting of the following tools

Github: Versioning and code sharing
Neptune: logging and visualisation
Kaggle API: dataset upload/ download
GCP: data storage

So in practice we downloaded preprocessed data from google storage, trained our models using pytorch lightning where we logged with neptune and uploaded the latest version of our git repo and model weights to a kaggle dataset to use it in our inference kernel. This allowed us to experiment and iterate quickly.

We are planning to release our code on github soon after some clean up.

#### Architectures

Our ensemble consists of 7 models using the following backbones available in the timm repository. Instead of much augmentation we trained our models on different image scales using albumentations.

- 2x seresnext101 - SmallMaxSize(512) -> RandomCrop(448,448)
- 1x seresnext101 - Resize(686,686) -> RandomCrop(568,568)
- 1x b3 - LongestMaxSize(512) -> PadIfNeeded -> RandomCrop(448,448)
- 1x b3 - LongestMaxSize(664) -> PadIfNeeded -> RandomCrop(600,600)
- 1x resnet152 - Resize(544,672) -> RandomCrop(512,512)
- 1x res2net101 - Resize(544,672) -> RandomCrop(512,512)

We normalize the images by the mean and std of the imagenet dataset before feeding them into a pretrained backbone. All models use GeM pooling for aggregating backbone outputs. We use a simple Linear(512) + BN + PReLU neck before feeding into an arc margin head with m ranging from 0.3 to 0.4 predicting one of the 81313 landmarks. We use the 512 dimensional output of the neck as the image embedding (= global descriptor) The following illustrates our setup for a SEResnext101 backbone.



#### Training strategy/ schedule

We train all our models on gldv2 clean data only. Each model is trained for 10 epochs with a cosine annealing scheduler having one warm-up epoch. We use SGD optimizer with maximum lr of 0.05 and weight decay of 1e-4 across all models.

#### Ranking post-processing

As previous editions of this competition have shown, properly ranking and re-ranking predictions is crucial to improve the GAP metric at hand that is sensitive to how landmarks and non-landmarks are ranked respectively. So one major aspect is to specifically penalize non-landmarks that constitute a large portion of the test set. We always tracked both overall GAP as well as landmark-only GAP separately and evaluated all ranking experiments on our validation set that resembled the test set quite well. There are quite different ways to approach this ranking problem, and different ways can lead to success, but here is what we found to work extremely well.

In the following graphic we visualize the main concepts of our ranking process. Test refers to the test set on the leaderboard, so the images we need to rank. Train refers to the candidate images we can use to determine the labels and the confidence. One important thing here is that we increased this set of images by all available images for the classes from gldv2_clean from gldv2_full. The 3rd place solution [1] notes that extending the dataset to include also these images improves their training, but what we found is that this is even more useful to be included in the inference process, which makes sense as there are more images to choose from. This also worked well on CV which is how we found it. And finally, Non-landmark includes all non-landmark images from the gldv2 test set. We then calculate all-pairs similarity between all of these sets. A measures the similarity to all available landmarks and their confidence. B measures the similarity of all train images to all non-landmark images and C does the same for the test images. 

The core idea is now to penalize A by B and C, so to penalize images that are similar to the non-landmark images. A is calculated between each test image and each train image. B and C are calculated by the mean similarity between the image and the top-5 (or 10) most similar non-landmark images.

Then, the first step is to penalize A_ij by B_j, pick the top-3 most similar images, and sum the confidence for the same label and then pick the highest. Afterwards, this score is further penalized by C_i. Actually, using both B and C for penalization is a bit redundant, and just using either of those brings good improvements, with just using B is better than just using C.



This whole procedure helps us to boost the actual landmarks to higher ranks and to penalize non-landmarks and rank them lower. Specifically B helps us to eliminate noise from the Train images. This gave both impressive boosts on CV and LB. 
One more important thing to note here is that when calculating cosine similarity between different sets, the similarity metric benefits from similarly scaled vectors. So what we do is that we fit a QuantileTransformer (other scalers work similarly well) on the test set, and apply them on the train and non-landmark datasets. This makes the scores way more stable and we assume that this also adjusts differently sized images better.

#### Blending

For blending our various models, we first l2-normalize them separately and concatenate them and apply above mentioned quantile transformer on each feature. We then calculate for each model separately the top 3 scores from above and then sum the same labels across all top 3 scores from all models and select the maximum label. For calculating C, we use the concatenated embeddings. This procedure is a bit more robust compared to just using the concatenated embeddings, but for simplicity one can also rely on that approach as it produces very similar rankings.
 
#### A word on local descriptors

We tried hard getting something out of local descriptors. For that we tried DELG as well as superpoint.
DELG: first we tried to port the pretrained DELG to pytorch but gave up after struggling for a day with tf1 and tf2.0 mixups in the original implementation. Instead we extracted local features using the tf implementation directly as done in the public baseline. 
Superpoint: We extracted local descriptors using the pretrained superpoint net, which is very fast.
We matched keypoints straight forward and applied RANSAC after. However the improvement even when using different image scales etc. was very small for both, DELG and Superpoint, and computational time for extracting and matching keypoints (especially when using DELG) was very high. Hence we did not use local descriptors in our final submissions.

#### What did not work

- Training together with gldv1
- Training together with gldv2 full
- Using index dataset 2019
- Using test set from 2019 stage1 
- Hyperbolic image embeddings
- Superglue
- Deformable Grid
- 1000 other things

Thanks for reading. Questions welcome.
 
[1] https://www.kaggle.com/c/landmark-recognition-2020/discussion/187757

update:
paper: https://arxiv.org/abs/2010.01650
code: https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place
