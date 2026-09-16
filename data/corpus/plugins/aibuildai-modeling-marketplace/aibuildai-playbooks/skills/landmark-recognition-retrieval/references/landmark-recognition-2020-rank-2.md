# 2nd place solution

Competition: landmark-recognition-2020
Rank: #2
Source: https://www.kaggle.com/c/landmark-recognition-2020/discussion/188299

Congratulations to all the winners, thanks to kaggle and Google for hosting Landmark Retrieval/Recognition competition. 

Here is my brief solution to this competition.

**Definition**
**GLD_v2c**(cleaned GLDv2), there are 1.6 million training images and 81k classes. All landmark test images belong to these classes.
**GLD_v2x**, in GLDv2, there are 3.2 million images belong to the 81k classes in GLD_v2c. I define these 3.2m images as GLD_v2x.
**query image**, the test image when submitting to kaggle or the image in validation set when validate locally.
**Index image set**, the images from train set, as described in Data page of this competition, “subset contains all of the training set images associated with the landmarks in the private test set”
**non-landmark image set**, sub set of 5000 non-landmark images form test set of GLDv2.

**Models to retrieve images**

It’s very important to search related images from index image set accurately. I trained the efficientnet B5, B6, B7 and Resnet152 models according to the first place solution of Landmark Retrieval 2020 competition from @keetar, https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176037. I will only describe the differences of my training strategy, please refer to his great solution for details.
1.Train 448x448 images from GLD_v2c for 5-6 epochs.
2.Finetune the model of step1 on 448x448 images from GLD_v2x
3.Finetune the model of step2 on 512x512 images from GLD_v2x for 5-6 epoch. I used GLD_v2x instead of GLD_v2 all.
4.640x640 for 3-5 epochs
5.736x736 for 2-3 epochs
 
**Loss function**: arcface instead of adacos loss.
**Optimizer**: SGD(0.01, momentum=0.9, decay=1e-5) 
**Inference**: extract embeddings by feeding 800x800 images.
**Validate set**: sample 200 images from GLDv2 test set as val set and all the ground truth images of Google Landmark Retrieval Competition 2019  as index dataset

After replacing the model in baseline kernel https://www.kaggle.com/camaskew/host-baseline-example from the host with trained efficientnet B7 model, 
the public and private score of B7 model are 0.5927/0.5582.

**Validation strategy for Landmark Recognition Task**
The val set part 1: the 1.3k landmark images from GLDv2 test set( exclude those not in 81k classes) .   
The val set part 2: sample 2.7k images from GLD_v2x-GLD_v2c.

The index image set for val set: all the images of related landmarks from GLD_v2c train set and sample some other images to get 200k images.

This strategy is quite stable during the whole competition, but unfortunately, the private test set distribution is a little different from my local CV and public test set. I should have used all the GLD-v2x images to generate index image set as many landmark images are not included in GLD-v2c. 


**SuperPoint+SuperGlue+pydegensac**
This combination is better than delf+kdtree+pydegensac.
The scored improved from 0.5927/0.5582 to 0.6146/0.5756, which can be top 10 on leaderboard.

**Post-Processing**

Although I force myself not pay to much efforts on post-processing, finding magic or reverse engineering, post-processing is so important in this competition that I had to spend a lot of time to analyse the model results and design rules on the validation set, I tried rules from winner solutions of last year, the following are the effective rules:
1.Search top 3 non-landmark images from no-landmark image set for query a image, if the similarity of top3 >0.3, then decrease the score of the query image.
2.If a landmark is predicted >20 times in the test set, then treat all the images of that landmark as non-landmarks.

As many features(ransac inliers, similarity to index images, similarity to non-landmark images...) can be used for determining whether an image is non-landmark or not, I developed a model which can be called re-rank model. As to re-rank, we can refer to this post:
https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159315

After post-processing, the score of efficientnet B7 model improved from 0.6146/0.5756 to 0.6797/0.6301 which can be top-3 on leaderboard.
**Ensemble**
My final scored was achieve by ensemble efficientnet B7, B6, B5, Resnet152 models, which is not a big improvement considering the complexity and computation resources.

repo:https://github.com/bestfitting/instance_level_recognition
[Update] with paper attached
