# brief summary of 2nd place

Competition: google-ai-open-images-visual-relationship-track
Rank: #2
Source: https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64651

Congrats to all the winners! and I'd like to thank Google and Kaggle for this interesting competition. 

This is My brief summary.

# Model 1: object detection (yolo)
I used yolo for object detection with following modification.

- Removed confidence term from loss function
- Added Class weight
- Class masking to forces only on labeled classes  
    But, did not mask the area where BB exists (BBs does not exist on same area)  
      But, masked for child class (BBs can exist on same area for child classes)   

# Model 2: visual relationship (InceptionResNetV 2)
     
Relation data(challenge-2018-train-vrd.csv) is mapped to all combinations of human labeled BBs(challenge-2018-train-vrd-bbox.csv) and the data is used as training data of model2.

[example]


### 2-1: relation 'is'
I made training data by mapping the ground truth BBOX(challenge-2018-train-vrd-bbox.csv)  to ground truth relational data(challenge-2018-train-vrd.csv), giving a target.  

- Objective: Material (wooden, plastic, ..., None). 
- Features: Cropped image, BBOX class, BB position / size etc.


Example:	

bbox data:

<pre>      image1,bbox1,Man
      image1,bbox2,Guiter
      image1,bbox3,Chair
</pre>


Relationship data:

<pre>      image1, bbox2, bbox2, Guitar, Wooden, is
      image1, bbox3, bbox3, Chair, Wooden, is
      image1, bbox1, bbox2, Man, Guitar, hold
</pre>

then, training data would be

<pre>      image1,bbox1,bbox1, Man,None,is  
      image1,bbox2,bbox2, Guiter,Wooden,is
      image1,bbox3,bbox3, Chair ,Wooden,is  
</pre>
　    here, target is None if the data is not in Relationship data.


### 2-2: Triplet Relationships  

First, I created all pairs of BBOX in the same image with some filter.
Then, I made training data by mapping the pairs to ground truth relational data, giving a target.

- Objective: Relationship (at, on, ..., None).   
- Features: Cropped image including two BBOX with box line, LabelName1, LabelName2, XCenter1, YCenter1, XCenter2, YCenter2, Size1, Size2, Aspect1, Aspect2, XCenterDiff, YCenterDiff, CenterDiff, XCenter, YCenter, IOU

Example:
bbox data :

<pre>      image1,bbox1,Man
      image1,bbox2,Guiter
      image1,bbox3,Chair
</pre>

Relationship data:

<pre>      image1,bbox1,bbox2,Man,Guiter,hold
</pre>


Then training data would be  

<pre>      image1,bbox1,bbox2,Man,Guiter,hold
      image1,bbox1,bbox3,Man,Chair,None
      image1,bbox2,bbox1,Guiter,Man,None
      image1,bbox2,bbox3,Guiter,Chair,None
      image1,bbox3,bbox1,Chair,Man,None
      image1,bbox3,bbox2,Chair,Guiter,None
</pre>
　  here, target is None if the data is not in Relationship data.






# Model 3: Score Prediction (Light GBM)
Relation data(challenge-2018-train-vrd.csv) is mapped to all combinations of predicted BBs(output of model1), and the data is used as training data of model3.

- Features: model1 output(BBOX classes, BBOX scores, BB positions / sizes etc.), and model2 output(Relationship and its probability)
- Objective:  whether the prediction of model2 is true or not

Model2 and model3 can be merged to one NN model.

# About yolo object function

The data set of this competition has the following three characteristics.

## (1), every images are not checked for all classes.
For example, there are cases where cats are not checked even if there is a cat in the image.
This causes false penalty if the model detects a cat during learning.

## (2), each object has a parent-child relationship.
For example, cat class is a child class of the Animal class, so a BBOX of cat is also a BBOX of Animal.

## (3), the 500 classes are not balanced.

To handle these better, I first deleted the confidence term from YOLO's objective function.
By deleting this term, class probability will be responsible for confidence.
Once class probability also have confidence role, this make it possible to mask unchecked classes.
By masking the cat class in the example above it becomes possible to eliminate the penalty when cat is detected by the model.

However, It is unlikely that different classes of boxes will be in the same area(same place, and same size, and same aspect).
So, I did not mask for the same area where some bbox exists.

But, there is an exception to this, as bbox of child class can exist in the same area(Animal can be a cat as well).
So I masked the child class of the corresponding BBOX.

In order to cope with an imbalance of (3), I removed same images.
But sampling images is not enough to balance the classes, so I introduced class weight.
