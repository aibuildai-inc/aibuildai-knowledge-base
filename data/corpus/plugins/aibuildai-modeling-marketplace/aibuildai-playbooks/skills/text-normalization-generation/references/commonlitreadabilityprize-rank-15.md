# 15th Place - Trust Your CV and Model Diversity!

Competition: commonlitreadabilityprize
Rank: #15
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/260800

## 15th Place Team “Trust Your CV” and Trust Diversity!

Thank you Kaggle and CommonLit for a fun and exciting competition. Thank you team for a great collaboration! @mobassir @ragnar123 @datafan07 @ivanaerlic @cdeotte Our team strategy was diversity of NLP transformer models because the train dataset was small and risk of overfitting was large. Besides diversity of backbones, our secret sauce was lots of creative model heads and creative training schedules including [RAPIDS cuml SVR][3] head which boosted CV LB +0.002!



## Congratulate Mobassir and Ragnar Becoming Competition Masters!

@mobassir and @ragnar123 have been sharing high quality notebooks and discussions with the Kaggle community for years. Their code and discussion sharing has personally helped me achieve gold in previous competitions! I’m excited to share in their current achievement of gold medal and becoming Kaggle Competition Masters. Please help me congratulate them. Great job guys!


## Introduction

Our team began with @mobassir and @ragnar123 who had achieved top 50 among 3000 teams. I was happy when they invited me to their team because I was struggling to get in the top 200 teams! As a team of three we decided diversity was the best approach in this competition. We invited @datafan07 and @ivanaerlic to join our team. These two Kagglers have demonstrated they can build creative diverse strong single models in their previous competitions. They succeeded once again! Thank you @datafan07 and @ivanaerlic , your models are fantastic!

## Trust Your CV

Every model we computed CV and LB. If both were good, we added the model to our ensemble. Our final ensemble had CV 0.452, Public LB 0.451, and Private LB 0.450. Notice that there was a strong connection between CV, Public LB, and Private LB.

## Mobassir - PyTorch - DeBERTa-Large
Configuration (CV 0.472 LB 0.463) : 

* Used single seed (63)
* Used self-attention head
* Used layerwise learning weight decay
* Trained for 7 epochs 
* Used learning rate 0.00003 for AdamW
* Used batch size and validation frequency = 8
* Used max_length = 300
* Used abhishek thakur’s bin based stratified5fold
* Tried SVR head but couldn’t improve cv furthermore because we think the self attention head was already trained very well

## Ragnar - TensorFlow - RoBERTa-Large

* CV 0.462 LB 0.459
* first two heads - cls token head, last hidden layer head 
* third head - SVR head
* two stage training with and without frozen backbone

## Ertugrul - PyTorch - Electra-Large, Funnel-Large, RoBERTa-Large

* CV 0.458 LB 0.454
* Huggingface 
* Sentence Transformers
* Bayesian Ridge head
* dropout = 0 (explained [here][2])

The main idea here was to fine-tune as many as strong transformer models using Huggingface Transformers then get their embeddings using the Sentence Transformers package. First we inspected many transformer models doing 5 fold cross validation. So we can see how they learn and generalize, then take these fine tuned models into sentence transformers to create sentence embeddings. Here we got last layer embeddings plus cls tokens, then concatenate them into one 2D array with 2048 (These can change but are similar most of the time for big models) features for each instance. Then built and trained Bayesian Ridge head where takes these embeddings as input and gives target scores as float. Actually it’s not that different from the model shared here by @datafan07. We believe these give decently generalized results but to get more accurate results there’s the second part.

In the second part the main idea was combining embedding based predictions with neural network based predictions. For testing this idea first we combined bayesian predictions with a public model. This gave us a great increase in lb score (~0.01) from 0.466 to 0.454 So we planned to combine these with the other team member models. It worked pretty well with our own models too, so at late stages we concentrated on finding optimal ratios of the combination using OOF scores.

## Ivan - PyTorch - Microsoft DeBERTa-Large, DeBERTa-Large-MNLI

* CV 0.464 LB 0.458
* first head - self attention head
* second head - Bayesian Ridge head
* dropout = 0 (explained [here][2])

## Public Notebook - PyTorch - RoBERTa-Base

Thank you @andretugan We used your RoBERTa-Base public notebook in our final ensemble
* CV 0.477 LB 0.467

## Chris - RAPIDS cuml SVR and Post Process

* Boost CV +0.002 LB +0.002
* We learned that during inference, if we extracted the final hidden layer activations (sentence embeddings) from our transformer models and trained a RAPIDS support vector regression model and then ensemble this model with the original model head we could boost all our models’ CV and LB. The TensorFlow models boosted LB +0.002 and most PyTorch models boosted LB +0.001.
* We also learned that the test data has a mean target that is 0.02 lower than the train target. Therefore if we subtract 0.02 from our final predictions this boosted LB +0.001

## Food For Thought
In the last few days we began exploring additional ideas. With more time we believe some could have boosted our CV LB scores. One example is an excerpt compare model described [here][1]. Another is pseudo labeling the test data during submission and then training another RoBERTa-Base during submission. Another is NLP augmentations. Another is using external data.

## Congratulations Top Teams - Thank you!
Congratulations to all top teams. Thank you for all your sharing during and after the competition. Thanks everyone for reading our solution writeup!

[1]: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257446
[2]: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/260729
[3]: https://docs.rapids.ai/api/cuml/stable/api.html#cuml.svm.SVR
