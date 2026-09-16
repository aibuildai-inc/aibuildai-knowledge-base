# 28th Place Solution

Competition: nbme-score-clinical-patient-notes
Rank: #28
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323031

Thanks Kaggle and the competition hosts for a well designed NLP competition.  This summarizes the effort of team Y3S and it was a great experience working with my teammates @sgalib @syhens and @smnomaan !  This is a writeup of our collective team effort. 

 Even though we were hit by the shakeup and dropped from our 11th place gold position, this was a great learning experience. 

# Data
We realized that Pseudo Labeling (PL) was going to be an important piece in the competition so we spent significant time getting good PLs using a 2 stage approach. The first set of PL were based on our best **deberta-v3-large** model. We trained a number of models on 5 folds using these PL and then used an ensemble of models to further improve the PL. We only used per fold models to generate PL to avoid leakage. This approach improved CV and LB significantly (+0.002). For example **roberta-large** was able to get to 0.890 LB with this 2 stage approach and 20% sampled PL. 

# Models

Our team worked on training different architectures with different training procedures including:
- Adversarial Training (AWP)
- LSTM head for certain models

The 5 fold mean CV for most of these models were 0.891x so I will include the LB scores

| Model | Public LB | Private LB |
| ---      | --- | ---- |
|  Derberta-v3-large| 0.891  | 0.892| 
|  Derberta-v2-xlarge| 0.891 | 0.890
|  Derberta-v3-base|  0.890 | 0.890
|  roberta-large|  0.890 |  0.889 |
|  longformer-large|  0.890  | 0.889| 
|  electra-large|  0.890  |0.890 | 

What is amazing is that we can even get **0.890** on the LB using a simple **deberta-v3-base** model !

# Post Processing
We added some custom post processing based on the error analysis of OOF prediction:
-  Dropping spans if they were less than minimum length for that feature num based on training data. 
-  Modify start positions based on if the predicted text is valid (not space etc.) 

Post processing gave us a `+0.001` boost in CV and LB. 

# Ensembles

We used the char level averaging similar to public approaches. We developed a faster version of ensembling which allowed us to ensemble 5-6 5-fold (25-30 total) base/large/xlarge models. Weights were optimized based on Optuna on OOF CV. 

# Final Selection
Our final selection was an ensemble of diverse models 

debv2-xl  + debv3-large (awp) + debv3-large (no awp)+ longformer-large (lstm head) + roberta-large + debv3-base  **Mean CV 0.8947**

# Things that did not work for us

- Different ensembling techniques like voting 
- Post Processing techniques like dropping spans if they were less than X % threshold of text similarity to all train data (including PL) - While this showed a good improvement in CV, it didnt work on LB. 


Looking forward to the next challenge!
