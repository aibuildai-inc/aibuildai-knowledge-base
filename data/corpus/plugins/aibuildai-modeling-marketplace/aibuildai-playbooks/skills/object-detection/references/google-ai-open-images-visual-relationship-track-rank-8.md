# 8th place solution [0.17 Private LB]

Competition: google-ai-open-images-visual-relationship-track
Rank: #8
Source: https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64631

Software
========

Windows 10 + Python 3.5 + Keras 2.2 + Keras-RetinaNet [0.4.1]: 
https://github.com/fizyr/keras-retinanet

Main approach
=============

In this task, I was strongly guided by my results and predictions from the task of Google AI Open Images - Object Detection Track (ODT) simply due to the fact that the "ITEM" classes are contained in the predictions from that task. Namely, 62 classes out of 500. It seems logical that the more accurately you know the positions of the objects you seek, the more accurately you can determine the type of interaction between them. Subsequently, I even trained a separate model for these 62 classes of objects using training boxes from ODT.

Statement of the task: two objects from the set to 62 classes are given on the image, as well as their interaction from 10 types:
RELATIONSHIP = ['at', 'on', 'holds', 'plays', 'interacts_with', 'wears', 'inside_of', 'under', 'hits', 'is']
In this case the type ‘is’ has a separate property - it is defined for one object and means the type of material, I called its attribute of the object.

In general, I split the problem into two independent parts, the Item + Attribute search and the second Item 1+ Item 2 + Relationship search.

Part 1 (Item + Attribute)
=========================

I trained two Retina models. The first one try to find for all possible pairs of Item + Attribute present in Train. In total, there were 42 such pairs. That is, each class is for example "Plastic chair", "Wooden chair", "Transparent cup, etc.". The main problem is that for some pairs there are very few samples. For example, for a Wooden Mug, only 5 images.
This model alone gives on LB ~ 0.05550

The second model looks only for the types of objects from the list
LABELS_ATTR = ['(made of) Leather', '(made of Textile', 'Plastic', 'Transparent', 'Wooden']
I wasn’t able to use this model to increase score. But it probably possible for some rare cases from previous model.

Part 2 (Item 1+ Item 2 + Relationship)
======================================

In this case, I trained the Retina model for 9 classes, which is looking for the fact of having some kind of relationship. 

RELATIONSHIP = ['at', 'on', 'holds', 'plays', 'interacts_with', 'wears', 'inside_of', 'under', 'hits']

Model tries to predict minimal rectangle containing two items forming some kind of relationship

For the prediction of Item1 and Item2, I took the best model from the ODT task.
To find the answer itself, a heuristic was invented. All found boxes for item1, item2 and relation were cut off at a certain threshold. Then there was a full search was searched for IOU between the boxes, if the Relation Box had an intersection with Item1 and Item2 and these Items are part of the allowed ones, then answer was added with confidence_score = (w1 * Conf_Item1 + w2 * Conf_Item2 + w3 * IOU1 + w4 * IOU2 + w5 * Conf_Relation) / (w1 + w2 + w3 + w4 + w5). The values ​​for wi were selected through scipy.minimize.

Then on the received boxes the modified NMS algorithm was started, which worked on two boxes at once. This model (Item 1+ Item 2 + Relationship) independently gave the score 0.15330 on LB.

Failed experiments
==================

In addition to the heuristics, I also prepared a LightGBM model with an AUC metric that tried to predict whether the predicted rectangles are valid or not on the base of the label, coordinates of the rectangles, areas of rectangles etc. Thus I tried to automatically cut off some non-existent cases, for example, when the first object for ‘under’ relationship is above the second, etc. And on validation this model was really better than my heuristic, but on LB it was for some reason worse. I think this is due to a small number of images in the validation - just slightly more than 5K or with some kind of error in the code. The best score was ~0.12 on LB.

Definitely here you have to dig in the direction of second-level models. Ideally, split the whole train set with KFold and use it to generate a variety of OOF features, for example rectangles from different Object Detectors, etc.

In general, the validation in this task has always been much more optimistic for me, but in general It worked in the right direction (improvement of validation improved LB). For example, validation for my best submission:
Rel mAP 0.263953 Rel Recall 0.619921 Phrases mAP 0.342934 Pred score: 0.366739
LB: 0.20109 - almost two times lower.

Problems of dataset
===================

 - A relatively small amount of data in the training set
 - Absence of validation and its difference from the validation from the ODT task
 - The “under” class is almost not represented in the training set (only 34 samples)
