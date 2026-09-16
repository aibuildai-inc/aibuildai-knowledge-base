# [Simplest Gold medal ?]11th solution (Ridge + Detoxify + LightGBM)

Competition: jigsaw-toxic-severity-rating
Rank: #11
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306228

# WARNING
First of all, congratulations to all winners. 
Though, really fortunately, I got the gold medal, 
**90%(or more?) of this result may be just lucky.**

Therefore, my solution is maybe not so interesting,special, and beneficial to you .

What I did are just 
- search additional data on Kaggle datasets
- Use some libraries to get toxicity of text
- make features and train simple LightGBM


If it's ok, then please read my solution (or my poem).



## 4 lines Overview
step1 : train ridge regressor  by tf-idf of a lot of datasets 
step2 : Toxicity prediction library(Detoxify)
step3 : convert validation_data.csv to rate of more toxic for train
step4 : Train LightGBM with these features and additional naive features

## Overview_image
[画像]




# step1 train ridge regressor  by tf-idf of a lot of datasets 

## 1-1 : reading ridge regression notebook
I started from [this ridge regression notebook] (https://www.kaggle.com/samarthagarwal23/mega-b-ridge-to-the-top-lb-0-85?scriptVersionId=80534222).

The notebook is
- To vectorize, It use tf-idf
- ridge regression with a lot of external data 
- brute force weighting the ridge results 
(It takes a lot of memory, so I can’t weighting not so many models )

After reading and executing this notebook, I thought
- The more external data is, the more diverse prediction I can do
  - I thought diverse predictors are needed because the score is averaged by many workers.   
- Weighting the ridge results by another method
  - Finally, I chose LightGBM for optimal weighting the ridge results.

## 1-2 : search additional data  
The notebook contains these data to train ridge regression
- ruddit data
  - https://www.kaggle.com/rajkumarl/ruddit-jigsaw-dataset
- previous jigsaw data
  - https://www.kaggle.com/julian3833/jigsaw-multilingual-toxic-comment-classification
  - https://www.kaggle.com/c/jigsaw-toxic-severity-rating

And I added 
- wikipedia data
  - https://www.kaggle.com/manishguptads/wikipedia-toxicity
- Dynamically Generated Hate Speech Dataset
  - https://www.kaggle.com/usharengaraju/dynamically-generated-hate-speech-dataset
- Malignant Comment Classification
  - https://www.kaggle.com/surekharamireddy/malignant-comment-classification

I used them to train ridge regressor and train LightGBM with the regressor's output.
(continued on step 4)



# step2 :  Call toxicity prediction library

I use [Detoxify] (https://github.com/unitaryai/detoxify) . It’s toxicity scoring library by Bert like models trained  with past zigsaw competitions.
And I found [offline execution notebook](https://www.kaggle.com/steubk/detoxify-quick-prediction-without-internet)

This library have three kinds of pretrained model (original,unbiased,multilingual)
I use all of these models.
 
I also use  these models to convert train text to features .
(continued on step 4)


# step3 : convert validation_data.csv to rate of more toxic for train
For train data, I use provided validation_data.csv.
1 problem is how to make the target value.

As long as I read competition overview, test data score is calculated  like this↓

[図2]
In other words, the score is the rate of more toxic. 

Then, to train, I did the same conversion to train data. I train my model to predict the score directly.

# step4: Train LightGBM with these features and additional naive features
Finally, I concat all of these features,train,and predict score.
And I added some naive features(length of text, words count, and so on)
Please see the image I attached the beginning of this notebook.




# result
public LB 0.79242
private LB 0.81062

**Thank you for reading ！**
If you have any question, I'll answer (if I can ...)
