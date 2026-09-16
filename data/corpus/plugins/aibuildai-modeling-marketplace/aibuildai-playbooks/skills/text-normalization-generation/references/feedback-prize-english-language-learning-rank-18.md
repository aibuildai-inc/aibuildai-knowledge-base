# 18th place solution

Competition: feedback-prize-english-language-learning
Rank: #18
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/370974

# 18 th place Solution

First of all, I would like to thank kaggle and the hosting staff for hosting such an interesting competition. And I would like to say congrats to all winner. I could not get the solo-gold, but I learned a lot from this competition.

# 1. Overview (Almost the same as others)

My solution is weight average ensemble with 21 models Nealder-mead with pseudo labeling. This is almost the same as other solutions. My originality is the replacement \n to [Chapter]. This improved the cv 0.0015. And in this discussion, I show the my CV and Private LB relationship in detail.

# 1.1 Configuration





# 1.2 Relationship of cv and lb 








# 1.3 cv and score in each model
  ※ pseudo1 and 2 are explained on next chapter
  ※ The order is the order with the best private lb score


| EXP | model             | pseudo1| pseudo2 | pseudo<br/> dataset  | kfold | cv      | private lb | public lb |
|-----|-------------------|---------------------|---------------------|----------------------|-------|---------|------------|-----------|
| 214 | deberta v3 large  | 〇                   | -                   | FB1 only             | 4     | 0.4467  | 0.4378     | 0.4404    |
| 123 | deberta v3 large  | 〇                   | -                   | FB1 <br/>+ Commonlit | 4     | 0.4467  | 0.4379     | 0.4402    |
| 200 | deberta v3 large  | -                   | 〇                   | FB1 only             | 4     | 0.4471  | 0.4381     | 0.4396    |
| 250 | deberta v3 large  | -                   | -                   | -                    | 15    | 0.4470  | 0.4382     | 0.4410    |
| 161 | deberta v3 large  | -                   | 〇                   | FB1 <br/>+ Commonlit | 4     | 0.4472  | 0.4383     | 0.4401    |
| 102 | deberta base      | 〇                   | -                   | FB1 <br/>+ Commonlit | 4     | 0.4483  | 0.4387     | 0.4386    |
| 201 | deberta base      | -                   | 〇                   | FB1 only             | 4     | 0.4503  | 0.4390     | 0.4399    |
| 163 | deberta base      | -                   | 〇                   | FB1 <br/>+ Commonlit | 4     | 0.4505  | 0.4390     | 0.4403    |
| 215 | deberta base      | 〇                   | -                   | FB1 only             | 4     | 0.4486  | 0.4391     | 0.4389    |
| 190 | deberta base      | -                   | -                   | -                    | 10    | 0.4508  | 0.4393     | 0.4393    |
| 777 | deberta v3 large | -                   | -                   | -                    | 10    | 0.4479  | 0.4398     | 0.4413    |
| 1   | deberta v3 large  | -                   | -                   | -                    | 4     | 0.4486  | 0.4400     | 0.4421    |
| 170 | deberta base      | -                   | -                   | -                    | 4     | 0.4517  | 0.4402     | 0.4401    |
| 20  | deberta v3 large  | -                   | -                   | -                    | 15    | 0.4485  | 0.4403     | 0.4413    |
| 18  | deberta base      | -                   | -                   | -                    | 4     | 0.4516  | 0.4405     | 0.4393    |
| 131 | deberta v1 large  | -                   | -                   | -                    | 4     | 0.4515  | 0.4413     | 0.4431    |
| 164 | electra-large     | -                   | 〇                   | FB1 <br/>+ Commonlit | 4     | 0.4544  | 0.4415     | 0.4455    |
| 216 | electra-large     | 〇                   | -                   | FB1 only             | 4     | 0.4507  | 0.4416     | 0.4454    |
| 202 | electra-large     | -                   | 〇                   | FB1 only             | 4     | 0.4550  | 0.4418     | 0.4457    |
| 114 | electra-large     | 〇                   | -                   | FB1 <br/>+ Commonlit | 4     | 0.4516  | 0.4419     | 0.4458    |
| 17  | deberta v3 large  | -                   | -                   | -                    | 4     | 0.4502  | 0.4419     | 0.4430    |
| 30  | deberta v1 xlarge | -                   | -                   | -                    | 4     | 0.4522  | 0.4420     | 0.4423    |
| 33  | deberta v2 xlarge | -                   | -                   | -                    | 4     | 0.4540  | 0.4425     | 0.4432    |
| 29  | electra-large     | -                   | -                   | -                    | 4     | 0.4546  | 0.4432     | 0.4494    |

# 2. What worked
## 2.1 High Impact

* Not using the lower (cv : 0.002 improvement)

At first, I converted the sentence to lower. But this was no good. In this competition, we need to predict the grammer etc... So Upper word may be important.


* Preprocess (cv 0.002 improvement)

As shown in other solution, target name concatenated with space separators and added by [SEP] (cv 0.0005 improvement). After that, I replaced further line breaks with [Chapter] (cv 0.0015 improvement). Here is a concrete example.

```
Cohesion Syntax Vocabulary Phraseology Grammar Conventions [SEP] I think that students would benefit from learning at home,because they wont have to change and get up early in the morning to shower and do there hair. taking only classes helps them because at there house they'll be pay more attention. they will be comfortable at home. [Chapter] The hardest part of school is getting ready. you wake up go brush your teeth and go to your closet and look at your cloths. 　…
```

* AWP (cv 0.001 improvement)
* Large k-fold
* Nealder-mead ensemble 

I used nealder-mead ensebmle in each target. This could jump the public lb from 600th to about 130 th at that time by changing from the mean ensemble.

* Pseudo labeling 1 (1 epoch pretrain + train.csv training)

At first, about the leakage of pseudo labeling, I have experienced in NBME competition. So I could avoid the leakage. [my solution on NBME](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/323156)

As shown in other discussions, at first, I trained the 5 models using the train.csv and inferred the the feedback 1 ( and commonlit) dataset in each. By ensembling it, the pseudo label data was made. At second, I pretrained the pseudo label data. At last, I re-trained the train.csv from the pretrained models. For example, this could make the cv 0.447060 of deberta v3 large model. 

* Pseudo labeling 2 (pseudo labeled data adding to the train.csv)

This is the simple pseudo labeling. I added the pseudo labeled data (as explained pseudo labeling 1) to the train.csv, and training it. I could make the cv 0.446743 of deberta v3 large model.

## 2.2 Small Impact

* Using CommonLit data for pseudo labeling

We were allowed to use Commonlit data [reference](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/348973#2047629). The ensemble improved the cv a little, but only slightly.

* Changing seed

Changing the seed and ensemble improved the cv a bit, but I think it was overfitting.


# 3. What didn't worked for me

* GBDT post-process with tf-idf etc.
* Replacement [MASK] in sentence.
* Cutmix
* adding LSTM to model
* Embedding SVR with fine-tuned models

# 4.Important ciations

* FB3 / Deberta-v3-base baseline [train] [link](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
* FB3 / Deberta-v3-base baseline [inference] [link](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference/notebook)
* (fgm)FB3 Deberta v3 base[train] [link](https://www.kaggle.com/code/mozattt/fgm-fb3-deberta-v3-base-train)


# 5.Thanks and Acknowledgements:
For various reasons, I started 11 days before the deadline. At this point, the various public notebooks were out and I was able to incorporate them into my pipeline and improve my score. @nakama and @Mozattt deserve my deepest thanks.

Also, I learned a lot from this competition, especially the Efficiency Prize Evaluation, which was very practical. In real work, it is often costly and time-consuming to prepare a GPU, and this initiative helped my work a lot. Finally, I would like to say thank you again to my host, kaggle, for organizing such a competition.

# 6. Team Members
chumajin
