# 14th place solution

Competition: data-science-bowl-2019
Rank: #14
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127221

First, I would like to thank the host, Kaggle, and everyone for this great competition. I would like to share my deepest gratitude to my teammates @alijs1 @johnpateha @kazanova . It was a really good time and joyful period for me.  

I will briefly brief our solution, and if I miss something, my teammates can add in their perspective.

## **FEATURE ENGINEERING**
Besides some popular features available in public kernels, we have some more custom features, such as ratio of good actions / all actions in Activity sessions, ratio of misclick or misdrag action / count actions (or session duration), count of some specific event codes since the previous Assessment session. The most interesting features, would be stats on the same assessment with regards to each data sample. This class of features helps tree models to converge quicker and reduce the importance of Assessment title.
As a separate solution, Marios (@kazanova) crafted his own feature set which took into consideration the train/test mismatch.  He can share more about this if needed.

## **DATA AUGMENTATION**
It is surprising that we did not realize the possibility to augment train data by a lot of test samples with true labels. Whenever a test installation_id (hereinafter referred to as “id”) has more than 0 prior assessment, we can trim the user history and make extra training samples. This augmentation helped us a lot in the blend.

## **MODELLING**
+ **Model 1**: Main Model for all assessments
+ **Model 2**: Five separate models for each assessment, then concat result.

Besides using all original data (17690 rows) as the main base model, we noticed that training 5 separate submodels for each type can give a boost if properly blended with the main base model. And since we also have the option of data augmentation, it results in 4 different training strategies in general. 

+ **Find threshold:** We use a simple optimizer to find threshold based on pure CV. People care too much about searching for right threshold, but we don’t. Instead we care more about modelling and ensembling, so thresholding would cast less effect.

+ **Train with sample weights**
We saw a significant LB boost if using appropriate sample weights in training. CV might not be boosted a lot, but LB is. We use the number of prior assessment as a criterion for assigning weight for each sample. The problem here is that we cannot naively use histograms of this criterion between train/test to calculate weights. The issue is that samples from the same id are much similar, so the effect of each individual sample in a single id should drop. For instance, if the ratio of 0-prior-assessment samples in train data is 1/4, and the ratio of 0-prior-assessment samples in test data is 1/2, then we cannot just simply assign weight=2 for all 0-prior-assessment samples in train, but a smaller value. In the end, we did not come up with a theoretically concrete strategy on how to get the weights, but just to roughly estimate it. We chose [1.65 , 1.09, 0.87, 0.77, 0.57, 0.47] as the weight for samples with 0-prior, 1-prior, 2-prior, 3-prior, 4-prior, and more-than-4-prior assessments, respectively.

## **ENSEMBLING**
+ **Blend by Classifier Logic**
We found a nice way to combine the main model result with 5-submodel result. We trained 3 simple classifiers with AUC loss:  A) classify between class 0/1, B) between class 1/2, and C) between class 2/3. Then we use the following custom logic to combine 2 float predictions of model 1 and model 2 to class label: 

| Abs(model1_int – model2_int) | model1_int | model2_int | Classifier | Result |
| --- | --- | --- | --- | --- |
|  0 | | |	| model1_int |
|1|	0 (or 1)|	1 (or 0)|	A &gt;= 0.2|1|
|1|	0 (or 1)|	1 (or 0)|	A &lt; 0.2	|0|
|1|	1 (or 2)	|2 (or 1)	|B &gt;= 0.5|	2|
|1	|1 (or 2)	|2 (or 1)	|B &lt; 0.5	|1|
|1|	2 (or 3)|	3 (or 2)|	C &gt;= 0.85|	3|
|1	|2 (or 3)|	3 (or 2)|	C &lt; 0.85	|2|
|&gt; 1	|	|	|	|(model1_int + model2_int) / 2|

+ **Stack**
Stacking also worked for us, both in CV and LB. As a result, we chose 1 final submission for the classifier logic, and the other 1 for stacking. For stacking, we tried 2 approaches: 4 stackers average, and extra-tree regressor. The latter performed better in CV and private LB, but we did not choose it and instead chose the blend of classifier logic + stack, which is bad in private LB.

## **WHAT WORKED IN PRIVATE LB BUT NOT PUBLIC LB**
- Blend by histogram matching (use prediction histogram of the best public LB submission to rectify private test predictions): very bad public LB, but very good private LB. 
- Extra Trees Regressor Stacking.
We would have finished "In The Money" zone if we chose this submission. However we don't regret.

## **WHAT DID NOT WORK**
-	Ranking average the predictions.
-	Pseudo label from unused train ids. Indeed, we observed high CV boost when using pseudo samples from unused train ids, but LB decreased. Instead we doubt we did not do it properly enough due to our code’s complexity.

## **WHAT WE DID NOT FINISH IN TIME**
-	We also developed an RNN model, which has CV 0.53x. This RNN takes as input two kinds of features: 1) the sequence of sessions as sequential data. Each session’s features are just count of different event codes. And 2) dense features which is same as those in LGB modelling. Indeed, this model can contribute in the blending, but we only finished this in the last day, so it was hard to combine into the code. We believe it would boost our score significantly.

## **WHAT DISTILLED IN MY MEMORY**
-	My first time to work with 3 great grandmasters in a big competition. It is my pleasure and great opportunity to learn from all of my teammates. Thanks a lot guys.
- I, personally feel happy with this result since we are one of the only 3 teams that can keep gold. Disappointment is overwhelmed by joy of lucky.
-	We sometimes felt that we a little bit hate kernel 😊 just because it run for 8 hours then failed at the end due to some minor error. However in the end I think it’s a good way for Kaggle competition: people cannot use black magic too much, and a concrete code base is needed, which makes room for coding skills enhancement for competitors. 
-	Combining solutions from team members is not a joke, especially if merging is late, like in our case (Evgeny and Marios only joined in the last week). It needs tons of efforts from all members. But in the end, if diversity of solutions is ensured one can expect a huge leap.
-	It is good that no extensive public sharing or any scandal appeared during this competition. 
-	Public LB/CV correlation is a mystery, which makes the competition more interesting.
-	Diversity is important, and is the key factor to avoid shake-up. We tried to bag training a lot, with lots of feature sets and models from each member. 
-  We have no concrete sign (for example, CV and public LB) to select our best private LB, so in general we don't regret the result too much. The gold position is somehow the result of our hard work and general sense of shakeup. So we will enjoy this gold medal a lot!

Thanks for reading, and hope you like this write-up. Our kernel is posted here. 
https://www.kaggle.com/khahuras/bowl-2201-a?scriptVersionId=27403894
