# [1st place solution]pretrained models and ensemble

Competition: imaterialist-challenge-furniture-2018
Rank: #1
Source: https://www.kaggle.com/c/imaterialist-challenge-furniture-2018/discussion/57951

First of all, we should thank competition sponsors and Kaggle for organizing a competition that is quite interesting. Besides,  all the competitors are quite excellent, maybe we are just a little lucky. Next, we want to briefly share our methods here.

We use pytorch to do all the training and testing, and our pretrained models are almost from the https://github.com/Cadene/pretrained-models.pytorch. Thanks for Cadene's sharing.

Most of our codes can be found here https://github.com/skrypka/imaterialist-furniture-2018,  just as Dowakin shared before, and we have updated it.

We totally have 12688 pictures for testing, for the missing ones, we just use the result in the sample.csv .

The models we choose are inceptionv4, densenet201, densenet161, dpn92, xception, inceptionrResNetv2, resnet152, senet154, renext101 and nasnet. The accuracy in the val set is 85% to 87%，except the nasnet which is quite hard to train and the accuracy is only 84%. 

For training I used SGD in most cases. The learning rate of the fully connected layer is ten times than others. The initial lr is 0.001 and switched to 0.0001 later.But for SEnet, it is hard to train with SGD, so I used Adam first. 

For testing, we use 12TTA = (normal+horizontal flip)*6 crops(full image, center, 4 corners) for each model and save the result. Finally we use gmean to get the final result  for the 12*10 12688*128 arrays. And the result is greatly improved  in this way. Apart from that, we notice the imbalance of the result. The test set should be 100 pics for each class, but the training set is not imbalanced. Thus the softmax will jugde by the training rate. We wrote a function to calibrate the probablity. You can see https://www.kaggle.com/dowakin/probability-calibration-0-005-to-lb/notebook.

We also add some weights to the 10 results when we do ensembing, which is useful to get a high score on the public leaderbord. However, for  the final submit we removed it just to avoid the overfitting in the 30% datas. It seems to be effective for the final score.
