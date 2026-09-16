# 1st solution

Competition: nbme-score-clinical-patient-notes
Rank: #1
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323095

First of all, thanks to competition organizers for hosting this competitions and great teammate(@ryuichigt). And thanks to the great notebooks and discussions, I learned a lot and was able to make it my final solution.

## Solution Summary

The following six models are blended to maximize cv. Each models use MLM, pseudo labeling, AWP, augmentation, and trained 5 out of 10 GroupStratifiedKFolds and all data, built our CV from the resultant 50% train data OOF. Final submit uses some weights of those six (fold 0, 1, 2, 3, 4, all) due to time limitations. [Notebook](https://www.kaggle.com/code/currypurin/nbme-final-cv-ver2?scriptVersionId=94661636)


|exp-num|Hugging Fase Model| cv | PubLB<br>(5/10folds+all) | PriLB<br>(5/10folds+all) | final submit<br>used weights|
|:---|:---|:---|:---|:---|:---|
|107|deberta-v3-large|0.8925|0.8931|0.8929|4/10folds+all|
|092|deberta-v2-xlarge|0.8914|0.8930|0.8911|3/10folds+all|
|106|deberta-large|0.8913|0.8922|0.8925|2/10folds+all|
|102|deberta-v2-xlarge|0.8925|0.8928|0.8940|3/10folds+all|
|101|deberta-large|0.8907|0.8918|0.8934|4/10folds|
|104|deberta-v2-large|0.8921|0.8925|0.8920|2/10folds+all|

[NBME Model PipeLine 003]

## Things that worked.

* AWP
  * [Feedback Prize 1st Place Great Notebook](https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook) is very helpful.
  * trying several parameters, the following improved the cv
      * adv_lr: 1.0, adv_eps: 0.01
  * cv increase about 0.002
  * [my code](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/323095#1777969)

* [MLM](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/316172)
  * Pre-training is performed on 'patient_notes.csv' except for the data in 'train.csv'.
  * cv increase about 0.002

* pseudo labeling
  * Ratio, pseudo labeling data: 90%, train data: 10%
  * soft label
      * In my experiments, soft labels improved cv score better than hard labels.
      * We attempted to limit the data used for pseudo labeling, such as those with more than 0.95 predictive value for any one letter, but sampling randomly from the entire data set improved the cv.

* augmentation
  * Remove one sentence from the text.
  * Probability of augmentation, p=0.2
  * cvscore is almost unchanged. However, it seemed to be effective for this task and data, so we adopted it.
  * Pseudo-label data did not try augmentation.

* Auxiliary Target Learning
  * Add the beginning and end of the target, respectively. and learn with 3chanel.
      * channel_0: Normal target
      * channel_1: start: 1, else: 0
      * channel_2: end: 1, else: 0
  * The cv improvement is slight, but can be used for ensembles(blend) with pseudo labels. (I think taking the average of two pseudo labels would improve the accuracy of the pseudo label.)
  * Checking the PrivateLB scores, it appears that this attempt was effective.

## Not worked

* shuffle augmentation
  * augmentation shuffles sentences, makes cv worse.
* label smoothing loss 
* clip_grad_norm
  * I tried the following two things, but the cv got worse. Maybe we need to adjust some more parameters.
      * torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
      * torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

## postprocessing

* Remove leading and trailing spaces.
* this excellent notebook [Be aware of white space \[DeBERTa+RoBERTa\]](https://www.kaggle.com/code/junkoda/be-aware-of-white-space-deberta-roberta)very helpful.

## cv split
Up until midway through the competition, we were using 5fold GroupKFolds. The change from 10folds to GroupStratifiedKFolds has been a huge improvement for our team.


## Notebook
submit: https://www.kaggle.com/currypurin/nbme-final-submit-currypurin
MLM: https://www.kaggle.com/code/currypurin/nbme-mlm/

## Edit

Edit: cv score added
