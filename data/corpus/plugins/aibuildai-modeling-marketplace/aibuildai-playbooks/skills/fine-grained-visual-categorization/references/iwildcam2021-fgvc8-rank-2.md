# 2nd place solution

Competition: iwildcam2021-fgvc8
Rank: #2
Source: https://www.kaggle.com/c/iwildcam2021-fgvc8/discussion/245559

Our solution is based on a few EfficientNetB2 classifiers using a combination of under sampling and over sampling during training.

**Species Classifier**

For the animal classifier we used the provided MD3 detections as input for training. The megadetector cutoff confidence for training was 0.8. We optimized the images prior to the bounding box cropping using the CLAHE algorithm.  We trained a full EfficientNetB2 classifier with input size 256x256. Data augmentation is random crops, color jitter. We fiddled with CutMix and MixUp but didn't use that in the end. We used exponential moving average (EMA) on the weights during training.
Key to the train/ valid split was the use of locations (the integer). The validation set had to be split to contain and more or less the same distribution of species and the locations should be completely independent between train and valid.
For the imbalance problem we combined under and over sampling. The problem with oversampling is a higher chance of overfitting and the problem with undersamping is information loss.

**Geo Prior**

With the release of the Geo information we trained a separate (expert) EfficientNetB2 network for all of the countries in the datasets where the unknown location was bundled to a separate country.

**Inference**

For the inference we used the EfficientNetB2 classifier on the provided MD3 bounding boxes (confidence > 0.4) and discarded classifier results with a confidence > 0.7. When the prediction was an animal that wasn't in the trainset for that location we moved to the expert country networks and relied on that output. We didn't have all countries trained when the deadline passed, so there might be more to win here.
Since the bounding boxes are classified independently we also trained a classifier for the entire image as done in the previous year. The objective of this classifier was to act as a prior for especially herds. In the case of the multiple boxes we would like to prefer to classify the same species. We combined the full image classifier with the species classifier using weighted 0.15 for the image classifier and 0.85 for the species classifier.

**Counting**

For the counting we just used the max operator on the number of bounding boxes. The classifier output has been validated on 0.75. This made our counting algorithm really conservative. We only counted the animals we were absolutely certain. For the final submission we added mean absolute difference to the max, which improved the results a little.
The reason for this is the use of MCRMSE where miss-classification is a sever penalty. The other problem with the MCRMSE is in large herds weigh really heavy on the leaderboard scores. Since we undercount we took some risk there.



**Things that didn't make the cut (yet)**

We didn't use the inaturalist data. At all.

We have set up and auxiliary network where we use the provided metadata to try and act as a regularizer. We used circular encoding for GPS and for the time of day and month. We used this information together with the image data to train the species classifier. Unfortunately we didn't have sufficient resources to fully pursue and evaluate this direction.

We did however use the same trick to create an evaluation network. The principle was the same and we tried to predict the species based on only on the metadata provided. Surprisingly we got validation score greater than 80%. So we still believe this is route to investigate in the future.

The final thing we didn't do is assume that all animals in one image are the same. We remained conservative in our predictions. What a good option was to push the scores was to assume that all high confidence bounding boxes contain all the same animals. This pushed our scores (private and public) significantly, but just liked adding the mean absolute difference their was no justification for that. We didn't have or use the true count present in the sequences to optimize either of them. This is also the reason for not pursuing the tracking or re-identification based counting algorithms.


 **Lessons learned**

- Classifier is still the most important contributor in this challenge
- Therefore the train/validation/test split is really important
- Metadata is valuable (but we don't know yet how much)
- Cattle is the favorite species classifier output (after empty)
