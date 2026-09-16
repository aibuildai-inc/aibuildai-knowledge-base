# 6th place solution (Gaussian process regression (GPR))

Competition: commonlitreadabilityprize
Rank: #6
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258554

### Acknowledgement
First of all, we would like to thanks Kaggle & CommonLit for hosting this interesting competition, this has been an interesting journey for us! Also congratulations to all winning teams!

And special thanks to @rhtsingh, @maunish, and @andretugan for their knowledgeable and resourceful notebooks. Since our team is fairly new to transformers, leveraging the resources from public notebooks definitely helps in our journey!

### Summary
Our final submission consists of 9 transformer models, and our final solution is based on our 2nd-stage model Gaussian Process Regression (GPR) [1] which we use to correct overconfident transformer predictions due to the small training data. This can be summarized in two steps. 
1. Extract all the OOF embeddings from the last transformer layer (Attention-head) for all 9 models and concatenate all of them.
2. Train a GPR model using the concatenated embeddings. 

### Architecture for final solution



### External data
We scraped data from the “url_legal” information which is available in the training data and only use those with licenses of “CC BY” and “CC BY-SA”.

#### Transformers model
Like every other team, we first started out fine-tuning transformers model and focus on increasing the single model LB score by training different transformers for diversity. Since our goal is to use GPR to improve the transformer predictions, we need robust transformer models that can generate robust embeddings.

Our final submission consists of the following 9 transformer models and all models are trained using 5-fold stratified CV with attention head.
| Model | CV | Public | Private | Notes
| --- | --- | --- | --- | --- |
| roberta-large | 0.492 | 0.471 | 0.471| mlm on training set
| microsoft/deberta-large | 0.485 |0.474 | 0.476| mlm on training set
| xlnet-large-cased | 0.494 | 0.475 | 0.476| -
| deepset/roberta-large-squad2 | 0.488 | 0.464 | 0.467| -
| deepset/roberta-large-squad2 | 0.484 | 0.466 | 0.464 | mlm on train set and external data
| allenai/longformer-large-4096-finetuned-triviaqa | 0.489 | 0.467 | 0.47 | -
| valhalla/bart-large-finetuned-squadv1 | 0.471 | 0.462 | 0.466 |  mlm on train set and external data, remove dropout
| microsoft/deberta-large-mnli | 0.469 | 0.462 | 0.469 | mlm on train set and external data, remove dropout
| ahotrod/electra_large_discriminator_squad2_512 | 0.477 | 0.468 | 0.468 | remove dropout

The model selection is based on the individual LB score (preferably <= 0.475)

### Fine-tune
The following are the fine-tuned strategy for the transformer models
- Epochs: 5
- Batch Size: 8
- Optimizer: AdamW with SWA
- LR Scheduler: Linear learning rate scheduler with 6% warmup steps
- Learning rate: 2e-5
- Layer reinitialisation: last 3 transformer layers (reference: https://www.kaggle.com/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning)
Use attention head. (reference: https://www.kaggle.com/maunish/clrp-pytorch-roberta-finetune)
- Evaluating for every n steps: 10

### Gaussian Process Regression (GPR)
We are aware of the limited training samples ~2.8k, hence we immediately thought of Gaussian process regression (GPR). Given the constraint of limited training samples, using Bayesian techniques such as GPR could potentially help us make a better approximation/prediction. Since Bayesian/Probabilistic model are proven to performs relatively well in small setting, and it also has a strong notion of uncertainty. The uncertainty is important here as we know the distribution of the training and test data are different, and we do not want the transformer to make overconfident predictions, hence GPR alleviates overconfident predictions from transformers. **A NN with one hidden layer of infinite width is basically a Gaussian Processes.**

The below figure shows the deberta-large model predictions for 100 randomly sampled points. The green curve is the deberta-large predictions, and the red curve is the GPR predictions. We can see that deberta-large will try to fit all the points as closely as possible (Red circle), however, GPR will avoid overconfident predictions and will regress to the mean of nearby samples when it is not confident, thus providing reliable uncertainty estimation.



### Gaussian process regression (GPR) as 2nd-stage model
We use GPR to correct overconfident transformer predictions. Doing this allows us to further improved the single model score significantly by training a GPR model using all the OOF embeddings extracted from the last transformer layer (attention-head). 

The following submission shows that using deberta-large trained with pseudo-labeling (PL) can achieve a single private score of 0.459, but with GPR it can achieve 0.453 alone!
| Model | CV | Public | Private | GPR Public | GPR private | Notes
| --- | --- | --- | --- | --- | --- | ---
| deberta-large | 0.462 | 0.465 | <b>0.459</b> | 0.457 | <b>0.453</b> | train with pseudo labeling

Unfortunately, we didn't include any model trained with pseudo-labeling since many of our PL-trained models has bad single/ensemble LB score as compared to the model trained without pseudo-labeling.. 

This leads us to believe the LB has a different data distribution as compared to the training data, hence we use all the OOF embeddings for training and only validate the individual GPR model performance on LB.   

### Training GPR using all oof embeddings
From the transformer models section above, the 9 models were selected in terms of their individual LB scores and correlations for diversity. We extracted the attention head embeddings which is of size 1024 for each model, hence 1024\*9 embeddings for 9 models. We concatenate all 1024\*9 embeddings and train all samples using GPR model with an initial RBF kernel hyperparameters of lengthscale=3, variance=1, noise=1, and run for 1000 steps using stochastic variational inference (SVI). Since we are doing inference with GPR, we are able to learn the GPR/kernel hyperparameters directly from the embeddings itself with minimal fine-tuning.

Our final submission with GPR model achieved public 0.448, private 0.448

### What worked
<b>Dropout removal</b>
The `remove dropout` note in the table above refers to removing dropout in the transformer architecture, instead of the dropout placed in the output head. We were inspired by this <a href='https://www.kaggle.com/andretugan/lightweight-roberta-solution-in-pytorch'>kernel</a>. It is a very interesting idea to remove it as the default dropout of 0.1 is used in nearly every transformer model. Previously, we can see that the validation loss is very unstable, which is why a small evaluation interval is commonly used. But after removing dropout in the transformers, the val loss became much more steady, and the converged points were usually much better. The cv score improved by a large margin, but the public LB doesn't reflect the change accordingly. Later we found more and more people used this strategy, so we just adopted it in some of our models.

<b>Pseudo labelling</b>
We scraped the data from the same url in the training data and labelled them with an ensemble model which scored LB 0.450. We didn't train on the pseudo data first and finetune with the competition data. Instead, during each fold training, we only added the pseudo data which have the same id as the training data in that fold, to avoid potential data leakage. With this method, the CV score improved a lot but the public LB didn't agree. Though we didn't found data leakage in the cv, we abandoned the models trained with pseudo labelling. It turned out that it works well for the private set, and we should've trusted the cv more. The public test set is only ~600 samples so it doesn't reflect the data distribution in the private set well.

<b>Pretrained models</b>
We found that some of the models further pretrained on question answering datasets are better for finetuning in this competition such as `squad` and `triviaqa`. So it's always better to try different pretrained models when using transformer models.

<b>Gaussian process regression</b>
This is the most important ensembling/preprocessing/correction steps for our solution. All of our single transformer models improved significantly with gains from 0.002 - 0.01 after training a GPR model using the attention-head embedding.

### What didn't work
<b>Extreme data and post-processing</b>
We found that the data with the targets < -3 or > 1 are the ones that make the training unstable. There are only 131 of these samples in the training set. Their standard errors are also usually higher than the others. By removing these samples, the cv improved a lot. And we found that there are also some patterns in the targets of the extreme data that we can use for post-processing, only if we can detect these extreme samples. But we failed to build a model to achieve that.

<b>Using different heads</b>
We tried training transformers using different heads, e.g. cls, mean pooling, pooler, concatenate last 4/8/12 layers, mean max pooling. But attention head works best for us, hence we stick with attention-head for all our models.

<b>Different optimization strategies</b>
Initially, we also tried mixout regularization and multi-sample dropout, doing this stabilizes training, but the individual LB score is bad. And in the end, the transformers w/o dropout works better.

### Final words
We should have trusted our CV for the models trained with pseudo-labeling. Our best submission has public 0.452, private 0.446 which is achieved by using only 6 models trained with pseudo-labeling. Unfortunately, we didn't select that for our final submission since the public LB is bad..

References:
[1] https://pyro.ai/examples/gp.html
