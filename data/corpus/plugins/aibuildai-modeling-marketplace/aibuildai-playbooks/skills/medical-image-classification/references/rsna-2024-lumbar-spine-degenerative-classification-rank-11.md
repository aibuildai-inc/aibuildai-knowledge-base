# 11th place solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #11
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569

First of all, we would like to thank the organizers and the kaggle team for organizing this competition.
We are honored to have achieved results in this very interesting and challenging competition.

# 1. Overview


* Our solution consists of 2stages    
  * In the 1st stage, we estimate the slice index to be inferred for each series and crop the region of interest.
  * Classifying severity levels with models using crop images as input at the 2nd stage.
* Label classification builds an independent model for each condition and concatenates the output of each to generate the final submission.
* The classification model is an ensemble of 4~7 models per condition.
  * The weight of each model is the value that minimizes cv in nelder-mead.


# 2. Pipeline Details

Our pipeline consists of the following two independent pipelines.
1. YumeNeko Pipeline
    * Pipeline built primarily by @kashiwaba 
2. YNK Pipeline
    * Pipeline with preprocessing by @kurimats and modeling by @takashimanaoya and @yosukeyama 

Each pipeline generates its own predictions, and the final output is produced by ensembling these predictions using a weighted average.
We have posted the details of each of these in the comments section of this discussion, so please refer to each comment.
* [YumeNeko Pipeline detail](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569#3013770)
* [YNK Pipeline detail - kurimats Part](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569#3014719)
* [YNK Pipeline detail - Naoya Part](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569#3014910)
* [YNK Pipeline detail - YYama part](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569#3016707)

# 3. Score
* The method of splitting the folds differs for each pipeline, but in all pipelines, we used StratifiedKFold with either a 5-fold or 10-fold split.
* The final scores are as follows
  * CV
      * spinal canal stenosis (including any_severe_loss): 0.251
      * neural foraminal narrowing: 0.464
      * subarticular stenosis: 0.524
      * overall: 0.373
  * LB
      * public: 0.35
      * private: 0.41
