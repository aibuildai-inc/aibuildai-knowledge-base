# 34th Place Solution - Layerwise and Sampling

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #34
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160896

I want to thank the organizers for this amazing competition and Kaggle for providing v3 TPU where people without hardwares like myself were able to compete without much disadvantage. I also want to congratulate the winners. This was my first Kaggle competition and I learned a lot about SOTA NLP models/techniques through the process. It was tough at first, but it was exciting to see my ideas work and move up the leaderboard. 

## **Summary**
1. ensemble of XLM-R models
2. layer-wise learning rate/weight decay 
3. pseudo-labelling
4. sampling strategy 

## **Ensemble of XLM-R models**
For all of my models, I trained on the training data for 2 epochs using [CLS] token without any extra layers. I tried different heads on top but I saw a drop in the score for all of them. I believe this is because the last-layer of a pretrained deep NN is too specific for the previous task (MLM objective in our case). Adding more layers on top just made the training more unstable. 

I further finetuned on the validation set for 2 epochs using 5-fold then averaged the oof predictions. This 5-fold CV AUC score was a good indicator of the public LB score. I ensembled only XLM-R models because other multilingual models (like XLM-17, m-BERT) were not very useful. I also trained each model 3 times and averaged the predictions for stability. 

Mainly, I trained 
1. XLM-R-large english only model (~0.9356 on public LB)
2. XLM-R-large translated only model - pretrained more with MLM using test data for 4 epochs (~0.9471 on public LB)
3. XLM-R-large translated only model with last 50 (~0.9449 on public LB)

## **Layer-Wise Learning Rate/Weight Decay**
I tried two different layer-wise strategies: layer-wise learning rate and layer-wise weight decay. I saw that layer-wise learning rate was effective in finetuning BERT in this article  https://arxiv.org/abs/1905.05583 and found it to be very useful for this competition as well. From this result, I also tried out a variant of this layerwise training strategy where I regularized the model more in the lower layers. I found that this worked much better than the layer-wise learning rate (about ~0.008 boost in public LB). It works just like the layer-wise learning rate strategy but instead of multiplying the learning rate by the layerwise coefficient alpha^(total_num_layers-layer_number) (where layer_number = 0 for the first layer), you multiply the weight decay rate by the coefficient alpha^layer_number. Thus, you are regularizing the lower layers more than the upper layers. This made sense because a wide exploration of the parameter space was needed for the upper layers in order to adapt to the new objective while the lower layers didnt need to change much. I used 0.01 for the weight decay and 0.99 for the layer-wise coefficient (alpha). Layer-wise learning rate strategy could actually be better but it was very sensitive to hyperparameter choices, so with limited TPU time I worked with layer-wise weight decay. 

##**Pseudo-labelling**
I used my best model's predictions on the test set as the pseudo-labels and included in the training set. Even using pseudo-labels of a relatively worse model (scoring ~0.935) was effective. It was interesting to observe the instability of the XLM-R-large model where switching only the pseudo-labels had resulted in diverging of the model. I had to lower the learning rate from 1e-5 to 6e-6 in order to converge.

##**Sampling-strategy**
 I downsampled training data to 1:1 for all models. For the XLM-R translated only models, I mixed yandex and google translation and saw a little boost in local CV and public LB. I also made sure that when sampling for non-toxic there were no duplicates across different languages for non-toxic.

##**Other Ideas that Didn't Work**
1. text pre-processing
2. label-smoothing
3. extending the test dataset by translating back from english
4. making multiple predictions - predicting at val finetune epoch 1 + val finetune epoch 2 and combining
5. meta-learning 
6. embeddings - I tried using bilingual Attract-Repel embeddings

Lastly, I learned that strictly planning your ideas by ranking them by their importance was really crucial especially when working with only 30 hours of TPU time. I found myself spending majority of my TPU time trying to tweak the working model to make it better rather than trying more of other novel ideas: striking a good balance between Exploitation and Exploration is the key! It was quite sad to see one of my ideas (making expert models using monolingual transformer models) in the middle of my untried idea list in the solution for the 1st place solution, but I have learned my lesson :)
 
I want to thank the authors of these wonderful kernels for their work. I have learned a lot from them.
https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmroberta
https://www.kaggle.com/riblidezso/train-from-mlm-finetuned-xlm-roberta-large
https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta
