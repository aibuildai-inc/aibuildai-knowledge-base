# 3th place solution - I CAn'T believe I won

Competition: cat-in-the-dat
Rank: #3
Source: https://www.kaggle.com/c/cat-in-the-dat/discussion/122649

**Model**
Logistic regression
C=0.095, class_weight={0: 1, 1: 1.4}, tol=0.00001,
solver='liblinear', penalty='l2'


**Encoding**
binary: 0 and 1
nominal, month and day: one hot encoding
ordinal: ordinal encoding


**Unseen value handling (only for nom_7, nom_8 and nom_9)**
The features which are not found in both training and testing data will be grouped into one class, called "other". For nom_9, there are many sparse features (counts of the features are small). To prevent overfitting, we also group these sparse features into the "other" class describe above. A featurea in nom_9 with count &lt; 3 (a parameter needs to be tuned) will be treated as sparse features
