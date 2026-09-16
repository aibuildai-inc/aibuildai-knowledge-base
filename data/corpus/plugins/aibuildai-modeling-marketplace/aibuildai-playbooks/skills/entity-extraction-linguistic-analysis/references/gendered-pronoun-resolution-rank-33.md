# 33rd place simple solution [LB 0.26663]

Competition: gendered-pronoun-resolution
Rank: #33
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90338#latest-522335

Thanks everyone for the competition! This is my first NLP kaggle, so I'm quite happy with the results. Thanks my teammates, [ods.ai](ods.ai) and kaggle community for great ideas. 
Now I would like to tell you about our solution.
# model
As a baseline we took this [kernel](https://www.kaggle.com/gdoteof/pytorch-bert-baseline-wd-epochs-cnn-lstm), so text tokenization and model structure was similar, but we changed **Head** structure of our model. First thought was to make it more "smooth". Linear layers were 1024 * 3 -&gt; 1024 -&gt; ... -&gt; 1024 -&gt; 3 for BERT-large-cased. And we made it 1024 * 3 -&gt; 1024 -&gt; 512 -&gt; 64 -&gt; 3. And for BERT-base-cased 768 * 3 -&gt; 768 -&gt; 768 // 2 -&gt; 768 // 8 -&gt; 3. We also put Dropout(0.5) before each linear layer (and bn, relu, of course). 
In both cases we didn't use the last BERT output layer. For base it was -2 and for large it was -4.
# training process
One important step: we used cleared data, from [here](https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/81331#503094), because some samples were labeled wrongly and it could corrupted results.
We used 6 folds. On each model was trained with BertAdam optimizer(lr = 0.0001, weight_decay = 0.01) from [pytorch-BERT repo](https://github.com/huggingface/pytorch-pretrained-BERT). And ReduceLROnPlateau scheduler with patience = 5 and alpha = 2. When we had no score improvement in 15 epochs we stopped training. We also tried CosineAnnealingLR after with RMSprop, but it didn't improved the score.
CV is great for stacking, so we decided to build a second layer model, which was LGBM. We added both BERT-large and BERT-base + some features.
# features
For features we used distance and url (as in this [kernel](https://www.kaggle.com/chanhu/bert-score-layer-lb-0-475)) and some statistic features, like 'num words', 'num unique words', 'num chars', 'num stopwords', 'num words upper' and encoded pronoun (just by number, ex. 'She': 1, 'he': 2, etc.). And that's it!
# LGBM
For parameters search i used a powerful [hyperopt](https://github.com/hyperopt/hyperopt) framework. This is very useful and effective for randomized search.
# fine-tuning BERT
We also tried BERT fine-tuning, like in examples from this [example ](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/lm_finetuning), but it didn't seem to work, the score was worse. So if you succeeded in BERT fine-tuning, please tell in comments how you did it.

Thanks for reading! 🥈
