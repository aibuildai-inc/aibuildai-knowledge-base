# Solution description (11th place)

Competition: inclusive-images-challenge
Rank: #11
Source: https://www.kaggle.com/c/inclusive-images-challenge/discussion/71732

## My solution - softmax  output for multi label classification

I noticed there are four key challenges of this competition: 

* 1. Labels are unbalanced in training data  - there are 800k instance of /m/01g317 , and only 1 instance of /m/0266h

* 2. The label distribution in training data and test data are very different.

* 3. There are large amount of classes/labels (7178 classes in classes-trainable.csv, but there are no training data for 6 classes,  so total 7172 classes)

* 4. It is not allowed to use pretrained model.

I tried following approaches to handle the above challenges:

## 1. Weighted sampling

To make training labels more balanced. I developed a sampler inspired by this post ( https://www.sebastiansylvan.com/post/importancesampling/ ), the original code is very slow to sample large amount of samples.  I made some changes to make it able to sample 100k samples in a few seconds. 

## 2. Label weights 

At the beginning of the competition, I used sigmoid activation function and binary cross entropy loss to train models for this multi label problem, but the loss and f2 score stop to improve quickly,  so I tried to put more weights on rare labels and positive labels during training.

* Class weights (not working well for me)

The rare labels should be put more weights, I calculate class weights per label counts, and calculate loss per class weights,  but this did not give me obvious improvement.

* Put more weights on positive labels (a big boost)

There are total 7172 classes, each image normally has only 1-10 labels,  the default binary cross entropy loss calculate all labels equally, I would like to put larger weights on positive labels to give the model a stronger signal on positive labels,  it can be done with pytorch simply as following:

```
def weighted_bce(args, outputs, targets):
    w = targets*args.pos_weight + 1  # default value of args.pos_weight is 20

    bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, w)
    return bce_loss 
```

This gave me a big boost on f2 score.

##  3.  Softmax output with fixed threshold 0.04 for all classes
I assumed that stage 2 test label distribution would be very different than stage 1, and tuning threshold for stage 1 tuning label would not work for stage 2. So I did not tune threshold for stage1 tuning labels.
Typically we use sigmoid activation on output for multi-label classification, but in this competition, I tried softmax activation on model ouputs, then use a fixed threshold 0.04 for all classes to generate prediction, for a same model, this method make the prediction can be better generalized to test data with a different data distribution.
I also tried sigmoid with tuning threshold for stage1 test label, which gave me around 0.48 LB at stage 1, but I did not use the approach at the end of stage 1 because I did not think that would work for stage 2.

## 4. Calculate Open Images mean and std
I calculated the entire training data mean and std, use them to normalize training batch.

## 5. Ensembling
I used two models for ensembling, since no pretrained model is allowed, it is very slow to train a single model, I trained each model for around two weeks on a single P100 GPU.  Use reduce on plateau learning rate scheduler first and then cosine annealing.
