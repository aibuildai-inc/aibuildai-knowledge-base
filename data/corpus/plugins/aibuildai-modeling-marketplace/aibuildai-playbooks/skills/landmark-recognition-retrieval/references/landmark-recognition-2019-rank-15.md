# 15th place Rapids.ai solution

Competition: landmark-recognition-2019
Rank: #15
Source: https://www.kaggle.com/c/landmark-recognition-2019/discussion/94308#latest-544734

Congratulations to winners. I'm by no means an expert in computer vision but I did my best for this competition. Here is a brief version of my solution and more code sharing will be done later.

**The major takeaway: please check out the cuml's KNN:**
https://github.com/rapidsai/notebooks/blob/branch-0.8/cuml/knn_demo.ipynb It is just so much faster. It just took **less than 5 mins** to find 5 neighbors of 110K test images from 4M train images.

The challenge of this competition is the large data size and number of classes. Given 200K classes and 4 million images with extremely uneven distribution, a simple classification model is difficult to make progress. My best classification model only yields 0.04 GAP, far less than the 0.19+ with the following pipeline:

1. Finetune a resnet152 and a densnet 161 for the most popular 10k landmarks classification. The validation accuracy is about 0.3.

2. Extract bottleneck layers, which are 512 features in my case. And apply a KNN to find K=5 nearest images in train given a test image based on these 512 features.

3. Repeat step 2 for different checkpoints of the two CNN models and aggregate the their KNN results. I used 10 saved CNN models to get 10 KNN models. At this point, each image in test has 10*5 = 50 candidate images from train. There are of course duplicates in these candidates.

4. Make it a recommendation problem by predicting a pair: a test image and a candidate train class. The target to predict is binary: if the test image is from the class than the target is 1.

5. Build features for such pair, like minimum distance from the test image to the images of a class, popularity of the class, so on so forth. I built 15 features in the end.

6. Run a xgb model and a NN model for the above data and average their predictions. For each test image, select the highest prob class.   

To be continued!
