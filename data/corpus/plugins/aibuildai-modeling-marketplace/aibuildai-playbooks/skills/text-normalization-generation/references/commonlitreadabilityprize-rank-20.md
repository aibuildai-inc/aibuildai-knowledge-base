# 20th place solution

Competition: commonlitreadabilityprize
Rank: #20
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257281

First of all I would like to thank the **CommonLit** Team and **Kaggle** for putting together an interesting challenge that allowed for the exploration of state-of-the-art NLP approaches for this challenge.  The small size of the training data and the variance in the labeling of the target by reviewers, made it an interesting challenge. 

# Approach at a high level: 
Our team **Force Awakens**   merged 1 month before the competition deadline and we found out that we had taken 2 different approaches to the challenge which worked quite well in our favor. 

- Regression Approach - I will outline the regression approach which is what a lot of the public kernels are using as well. 
- Pairs Model Approach - Discussed [here](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257360) This was pretty unique and something I haven't seen discussed in the forums. 

The regression approach by itself scored `0.454` on public LB and `0.457` on private LB. Combined with the pairs model approach which also scored `0.454` on public, `0.454` on private, our overall best score was `0.450` on public LB and `0.452` on private.  


# Regression Approach:
We trained many different architectures including base/large/x-large. Similar to what many people observed, we found that the large models worked best for this dataset and target distribution. We tried a lot (and I mean a lot!) of architectures including but not limited to *Roberta/Electra/Funnel/Deberta/StructBERT/BERT/XLNET/BART/T5/AlBERT//Luke/DistilBERT* etc. 
	The models that performed best on the regression task were:
		- **Roberta-large** - Public `LB 0.464` 
		- **Funnel-large** - Public `LB 0.465`
		- **Deberta-large**- Public `LB 0.466` 
		-  **Electra-large** - Public `LB 0.468` 
		- **Roberta-base** - Public `LB 0.467` 
			
 **Training approach**:
- This is where we spent a lot of time trying out different learning rate schedulers, loss functions, different embeddings of BERT models and custom heads on top of the embedding layers. 
- What worked best for us for the regression approach was using the CLS token and then using a custom head with 2 linear layers and GELU activation. 
```
		self.whole_head = nn.Sequential(OrderedDict([
            ('dropout0', nn.Dropout(args.use_dropout)),
            ('l1', nn.Linear(args.fc_size, 256)),
            ('act1', nn.GELU()),
            ('dropout1', nn.Dropout(args.use_dropout)),
            ('l2', nn.Linear(256, 1))
        ]))
```

- Training each model was a challenge in itself because there was no silver bullet of params that worked well for all architectures.  On top of that, the scores across different folds varied quite a bit so we took an approach where we could specify the fold/seed in the training script and re-trained poor performing folds again with different params like changing the LR, eval steps, use of MLM, dropout. 
- Pre-training the models on MLM task on competition data seemed to help but we never used it for all folds. Using the approach in the previous bullet point, we used MLM pretrained model on a few folds and that worked best on LB vs using it for all folds (possibly preventing overfitting)
-  Things that worked well:
			-  Using pre-trained MLM model on some folds
			-  Using a small learning rate 1e-5/2e-5 with cosine scheduler, 5 epochs and running eval intervals in each epoch often. 
			- Using differential learning rates for BERT layers
- Things that did not work:
			- Data augmentation
			- Adding additional readability features either to last NN layer in BERT or using meta-learning like SVR/GBM based models. 
			- Reinitializing of last N BERT layers - Local CV's looked promising but this never resulted in better LB at least for us. 
			- Multisample dropout
			- Some Architectures like XLNET, ALBERT, T5,  never converged. 
			- Different BERT embeddings like BERT pooler or combination of mean/max pooler, last N hidden layers etc. 
-  Our final regression model was a simple blend of the 5 architectures listed above. We tried meta-learners like SVR/Ridge/GBM on top but they didn’t seem to help much. 

On a personal note, I learned a lot about fine-tuning BERT architectures with various innovative strategies and that is something that will definitely help in the future (Kaggle/Work) 

Special thanks to all my team mates @eduardopeynetti @ryches @maxjeblick and @jesucristo  for coming up with great approaches and ideas that helped improve our models. 

Also a shout out to @rhtsingh for sharing some great ideas which I am sure a lot of people have seen and benefited from.
