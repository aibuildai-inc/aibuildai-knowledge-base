# 8th place solution

Competition: feedback-prize-english-language-learning
Rank: #8
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369524

## Overview

Our team ensembled 26 models in the following table. (including knowledge distilled and pseudo labeled ones) Ridge regression was used to integrate the prediction results of each model. Post-processing was applied to adjust the output values of the final ensemble's predictions. For the single model, the following techniques contributed to score improvement.

- Knowledge distillation and Pseudo labeling
- Preprocessing - replacing “\n\n” with [PARAGRAPH]
- AWP
- Change the max len for each epoch


| &nbsp; &nbsp; &nbsp; | Model                           | CV / LB            | Description                                                                                                                                                                          |
| --- | ------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1  | Deberta v3 base                 | 0.448| 512 Max length , Pseudo labels inferred from the tensorflow Deberta v3 base (LLRD technique) , and those plus competition data trained on the deberta v3 base (freezing+meanpooling) |
| 2  | Deberta v3 large                | 0.448              | Same as above , deberta v3 large trained on meanpooling + freezing                                                                                                                   |
| 3  | Deberta v3 small                | 0.447              | Knowledge distilled model                                                                                                                                                            |
| 4  | Deberta v2 xlarge               | 0.448 / 0.44| LSTM + mean pooling +freezing (Same pseudo label from tensorflow)                                                                                                                    |
| 5  | Deberta v3 large                | 0.4495 / 0.43      | Max len = 470, trained with decreasing max len - 768/512/470/470                                                                                                                     |
| 6  | Deberta v3 base                 | 0.4515 / 0.43      | Max len =470, Mean pooling, AWP, val\_steps = 250, reinit layers                                                                                                                     |
| 7  | Deberta xlarge                  | 0.4534 / 0.43      | CLS token, val steps =20                                                                                                                                                             |
| 8  | deberta v3 large                | 0.4495 / 0.43      | CLS Token, AWP, val steps = 250, max len 768                                                                                                                                         |
| 9  | Electra large                   | 0.4545             | Meanpool, LSTM                                                                                                                                                                       |
| 10 | deepset/deberta-v3-bbase-squad2 | 0.4522             |                                                                                                                                                                                      |
| 11 | Luke Large                      | 0.4551             | Max len 512, Mean pooling, bidirectional-LSTM                                                                                                                                        |
| 12 | Deberta v3 large                | 0.4467             |                                                                                                                                                                                      |
| 13 | Deberta v3 base                 | 0.4471             | AutoModelForTokenClassification, last 4 layer concatenate, mean pooling,max\_len=768                                                                                                 |
| 14 | Deberta v3 large                | 0.4469             | PL Model, 2 seed averaged, CLS Token                                                                                                                                                 |
| 16 | Cocolm large                    | 0.4568             | No \[PARAGRAPH\], CLS Token, AWP                                                                                                                                                     |
| 17 | GPT2 Medium                     | 0.4648             | max\_len=1024, Mean pooling, SWA                                                                                                                                                     |
| 18 | Longformer Large                | 0.4592             | max\_len=1536, Mean pooling                                                                                                                                                          |
| 19 | Bart Large Squad                | 0.4649             | max\_len=1024, Mean pooling                                                                                                                                                          |
| 20 | Distill Bart CNN                | 0.4607             | max\_len=1024, Mean pooling                                                                                                                                                          |
| 21 | OPT-350M                        | 0.4639             | max\_len=1536, Mean pooling                                                                                                                                                          |
| 22 | Bart Base                       | 0.4788             | max\_len=1024, Mean pooling, SWA                                                                                                                                                     |
| 23 | T5 Large                        | 0.4717             | max\_len=1024, Mean pooling                                                                                                                                                          |
| 24 | Deberta v3 base                 | 0.4490             | max\_len=512                                                                                                                                                                         |
| 25 | Deberta v3 base                 | 0.4484             | max\_len =384                                                                                                                                                                        |
| 26 | Deberta v3 large                | 0.4498             | max\_len=1429, Mean pooling, Pseudo Labeling                                                                                                                                         |



## What worked ? 

- **Preprocessing:** - replacing “\n\n” with [PARAGRAPH]  There are about 17.5K instances of \n\n in the dataset. Deberta v3 large tokenizer simply ignores these. So we decided to create one additional special token by replacing \n\n with [PARAGRAPH]. Also resized embeddings to take care of this one extra token. Just this change moved deberta v3 large from 0.44 to 0.43.

- **Knowledge Distillation:** We had one deberta v3 x small model with knowledge distillation. 
We decided to distill knowledge i.e. (get OOF data)  from an ensemble of 9 models and utilize these to train the model. This is only utilizing current competition data not previous feedback data.

- **Pseudo labels (PL)**  
This was challenging to determine if pseudo labels are working or not. We tried generating labels with a few different models to determine if it’s working. Eventually we created two seed deberta v3 large models which seemed more stable than trying to determine with single deberta v3 large. Some of the models in our ensemble were with PL. <br/>
We also did pseudo labeling inferred from the tensorflow Deberta v3 base model , which is trained with layer wise learning rate decay (LLRD) technique and used those PL inferred data into some of our models , which helps in improving the cv as well as LB.

- **Model Architecture:**
Just CLS token, Mean pooling, Mean pooling + CLS token, MaskAddedAttentionHead, MaskAddedAttentionHead + CLS token(AutoModel and AutoModelForTokenClassification) , LSTM

- **Training strategies:**
AWP worked, different max len per epoch, validation steps 20-250 were utilized for different models. Some of the models were trained using end of epoch validation as well. 
CV was calculated with 5 folds models.

- **Diversity in ensemble:** 
Folds - By Abhishek - https://www.kaggle.com/code/abhishek/multi-label-stratified-folds  <br/>
We had ensemble of total 27 models (runtime ~7 hrs)  - 7 Deberta v3 large models, 5 Deberta v3 base, Deberta v2 xlarge, Deberta xlarge, Deberta v3 small, Luke Large, Electra, Cocolm, Deberta v3 base squad, GPT2 medium, Longformer Large, Bart Large Squad, Distill Bart CNN, OPT 350M, Bart base, T5 large

- **Ridge Stacking**
We used Ridge for stacking our models, there were 2 deberta large (two seed models) which we combined so that generated only 1 ridge coefficient and it helped in ensemble. 


- **Post processing**
We used “Nelder-mead” https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html  for post processing - divided the predictions into 20 bins and multiplied the resulting array with predictions. 
This post processing is based on commonlit's 2nd place solution method.
https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328

Our final ensemble is here - https://www.kaggle.com/code/aman1391/fb3-8th-place-solution-ensemble-27-models?scriptVersionId=112097448

## Ideas that didn’t work
- **Mask Language modeling:** - Tried many different ways for MLM to work, unsuccessful attempts. Tried to match similarity between FB3 data and previous feedback data for MLM which also didn’t work. 

- **Other models:** - Various other models from HuggingFace didn’t work. A few did work however it wouldn’t work with our ensemble so they were unutilized.

- **Different loss functions:** - Focal Loss, Ordinal regression from public kernel, Asymmetric loss - Asymmetric loss did work with a few ensembles but it wasn’t part of our best submission. 

- **Ensemble with weights:** - In the last 2 days, we tried ensemble with weight tuning but Ridge did give us better results. This needed more tuning or work for us to feel confident in selecting a submission. 

- Many different model architectures, hyperparameters, token dropout, 4 fold model, dividing folds by number of tokens. 

## Other
- **Compute:**  A100s, 4090 (3090 and Colab Pro+ prior to 4090) 

- Please see this post for a single model (average 2 seeds) gold solution - https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369368 and corresponding [training code](https://github.com/rashmibanthia/feedback3) 


## Important Citations
- [https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
@yasufuminakama. Almost all of our scripts are based on this 
- [https://www.kaggle.com/code/vslaykovsky/lb-0-43-ensemble-of-top-solutions](https://www.kaggle.com/code/vslaykovsky/lb-0-43-ensemble-of-top-solutions)
@vslaykovsky This notebook made our ensemble easier   
- [https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/366525#2038773](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/366525#2038773)
Kelvin  @xyzdivergence - for sharing ideas that worked -  different max len in ensemble
- [https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328)
Post processing is based on commonlit's 2nd place solution method
- [https://www.kaggle.com/code/electro/deberta-layerwiselr-lastlayerreinit-tensorflow](https://www.kaggle.com/code/electro/deberta-layerwiselr-lastlayerreinit-tensorflow)
@electro We retrained above model on our folds and created pseudo labels

## Thanks and Acknowledgements

Thanks to our hosts for hosting the competition and to teammates. Thanks also to the Kaggler's for sharing their helpful notebooks and discussions.

Thank you and Congratulations  to my team mates - @aman1391, @kanbehmw, @fightingmuscle

**Team Members**
@rashmibanthia
@aman1391
@kanbehmw
@fightingmuscle
