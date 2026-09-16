# 6th place solution

Competition: youtube8m-2019
Rank: #6
Source: https://www.kaggle.com/c/youtube8m-2019/discussion/112403

Thanks organizers for hosting this competition! Congratulations for all the top teams! I hope that you have enjoyed this challenge!  🎉🎉 🎉  With [@daokouer ](https://www.kaggle.com/daokouer) we had a great time exploring the interesting topic.

Here we will present briefly our solution for this challenge. For this temporal localization problem , we regarded it as video segment classification problem. We trained our models based on video segment feature with or without context information and predictions were made for each segments.

## Models
We trained two types of the models: sequence model and frame level model. These two types of models made their decisions based on different parts of input.

### Sequence Modeling Model
We used Transformer and BiGRU as our sequence models. The whole video feature was taken as input and predictions were made for each five frames. For sequence models we believed that they focused on long-term temporal dependency. 

### Frame Level Model
NeXtVLAD were used as the frame level model. The frame level took exact five frames as input and output one prediction. We believed that with limited reception field, the NeXtVLAD focused more on the static feature of the segment.

## Model Pre-train
Models above had a great number of parameters to learn but we only had few segment level labels. We thus wanted to take advantage of the huge training data with video label. We used EM-like process to make use of the training set. We initialized our model `f` using the same method as the training process in the official baseline code. During E-step, we estimated the segment label of the training set using model `f`. During M-step, we trained a new model `f` using the generated label and fine-tuned it  on the segment label. We operated two EM iterations in our experiments.   
We also tried several multi-instance learning methods to make use of the video label, e.g. we performed max pooling over segment predictions and took it as video prediction to calculate the loss, etc. But we did not find a MIL method that outperformed the EM-like method. 

Please refer to workshop submission for detailed solution.
