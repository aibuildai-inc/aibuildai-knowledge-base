# 7th Place Solution Summary - anokas

Competition: google-ai-open-images-visual-relationship-track
Rank: #7
Source: https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64630

Congratulations to the winners and thanks to Google AI for hosting this fun and interesting competition. My solution was relatively simple, taking a submission file from the object detection competition as an input and building some heuristic models based off bounding boxes.

**Object Detection**

For object detection bounding boxes I used Google’s [pre-trained Faster R-CNN model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) trained on V2 of the Open Images dataset - the labels they were trained on were actually a superset of the labels in this dataset so it was just a case of mapping the labels and filtering out ones we weren’t interested in. After expanding the label hierarchy, this model scored about 0.37 in the object detection competition. I also predicted on flipped images and combined the predictions, netting me **0.375** overall.

One thing that led to a huge increase in score in object detection was to not threshold the predictions and leave the low confidence predictions in the submission file. Because of the way average precision works, you cannot be penalised for adding additional false positives with a lower confidence than all your other predictions, however you can still improve your recall if you find additional objects that weren’t previously detected.

**Visual Relationships**

This challenge can actually be split up into two subproblems. The first subproblem involves the `is` label (for example `chair is wooden`), and in fact only involves a single object and an attribute. This subproblem spans 57 relationships across 5 different descriptive attributes.

The second subproblem is indeed based around visual relationships, and involves a pair of objects as well as a proverb, such as `chair at table` - there are about 250 different triplets that occur within the dataset.

While the two types of problems have about equal occurences in the dataset, the triplet relationship detection has about 5x more distinct relationships - meaning it has about 5x more weight on the LB score, due to 80% of the metric being the mean of relationship-wise AP.

**Attribute Classification**

For this subproblem, I built a training dataset by getting the bounding boxes of all objects which could have a descriptive attribute from the bbox ground truth file. I then matched these objects with the boxes in the relationships ground truth file, which gave me a ground truth (for each object example, which attributes are present).

On this dataset, I then trained a single pre-trained DenseNet121 network on the cropped objects resized to 224x224, to predict the probability of all 5 classes - this achieved about 0.95-0.99 AUC on all the classes.

The pre-trained model is then used to predict the probability of the 5 attributes for all the applicable bounding boxes in the object detection challenge, with the final predicted probability as follows:

    P(chair is wooden) = P(bounding box is chair) * P(bounding box is wooden)

Where the `bounding box is chair` probability is the output of the object detection model.

**Triplet Relationships**

For the visual relationship triplets, I also built a dataset similarly to attribute classification - all potential pairs of bounding boxes of the two objects, and whether they have the given relationship.

This gave me around 250 training sets in total for the different triplets (of varying size).
For the first 100 relationships, I built a 5-fold XGBoost model on each one with the following features:

- % box 1 inside box 2, % box 2 inside box 1
- IoU (intersection over union between the boxes)
- Horizontal/vertical offset of the box centres
- Euclidean distance between the two box centres
- Euclidean distance normalised by the size of the boxes (zoom invariance)

And then using the models, the submitted probability of each relationship is for example:

    P(chair at table) = P(bounding box is chair) * P(bounding box is table) * P(chair at table XGBoost)

It seemed that building a separate model for each relationship worked well, as I noticed they each had very different feature importances (eg. sometimes the model was looking at the overlap, while for others the model was looking at the difference in height between the objects). Most classes also had &gt;0.95 AUC, showing the XGBoost model was very good at classifying whether a relationship existed.

For the other relationships (with a few hundred samples or less) I replaced the xgboost model with a simple prior - what proportion of pair occurences had the relationship in the training data. I tried using eg. linear models for these but didn’t see an improvement.

Overall, the solution runs from start to finish in under 24 hours on an i7+single GPU - I am happy with the performance given the simplicity. I observed a linear increase in performance in visual relationships given an improvement in the bounding boxes scores, so this solution could have potentially scored much higher if I had a better object detector! :)

I am curious to see how other competitors approached the problem, as this is quite a new problem for Kaggle.

\- anokas
