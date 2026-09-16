# 3rd place solution (0.818 LB): The Transformer

Competition: riiid-test-answer-prediction
Rank: #3
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209585

**2021-01-16 edit 2**: added missing linear layers after categorical embeddings + tag features to diagram
- **2021-01-16 edit 1**: the [source code](https://github.com/jamarju/riiid-acp-pub) is up.

Wow, what a ride this has been. First off, thanks @sohier @hoonpyotimjeon Kaggle, Riiid and everyone involved in setting up this challenging competition.

Congratulations to #1 and #2 @keetar and @mamasinkgs!!! We truly look forward to reading about your solutions!

Also huge thanks to my teammate @antorsae who joined me in the last stretch and without whose ideas, intuition and hardware I wouldn’t have gotten this far.

I was attracted to this competition by the relatively small dataset footprint compared to my other two previous competitions (deep fakes and RSNA pulmonary embolisms) but this ended up being much more resource intensive than I anticipated.

Our solution is a mixture of two Transformer models with carefully crafted attention, engineered features and a time-aware adaptive ensembling mechanism we nick-named “The Blindfolded Gunslinger”.



(“Blindfolded gunslinger” hand-drawn by @antorsae inspired by Red Dead Redemption 2)

# The Transformer



We use two transformers trained separately with 2.5% of the users held out for validation and sequences of 500 interactions. At train we simply split user stories in 500 non-overlapping interaction chunks and sample the chunks randomly.

* Transformer 1: 3+3 layers (encoder+decoder), no LayerNorms, [T-Fixup init] (http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf) see paper for reasons why we used it, ReLU activations, d_model=512
* Transformer 2: 4+4 layers, no LayerNorms, T-Fixup init, GELU activations, d_model=512

We feed both the encoder and the decoder ALL the features. We use learned features for continuous variables (simply projecting them to d_model) and for categorical variables we first map it to embeddings with low dimensionality and then project it also to d_model=512 (to avoid potential overfitting). We use an embedding bag for question tags.

To prevent the transformer from looking into the future we shift the encoder input including both questions + answers to the right, hide all the answer-specific features from the decoder (user_answer, answered_correctly, qhe, qet), ie. those that are not immediately available upon inferring the interaction and use the appropriate attention masks in all 3 attentions.

To our surprise the 3+3 model outperformed its bigger 4+4 brother even if we tried to finetune the latter at the final hours of the competition.

# Engineered features

This was a very rich dataset  but we found the following derived features helped the transformer converge faster and reach a higher AUROC score. A lot of them have been discussed in the forum:

* `qet`, `qhe`: these are the `prior_qet` and `prior_qhe` counterparts shifted upwards one container. This was probably discussed first by @doctorkael here: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/194184
* `tsli`: time since list interaction, AKA timestamp delta. Discussed in many threads.
* `clipped_tsli`: `tsli` clipped to 20 minutes. @claverru hinted at this in the Saint benchmark mega-thread: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/195632
* `ts_mod_1day`: `timestamp` modulus 1 day. This may reveal daily patterns such as the user being more / less attentive / tired in the mornings / after work, etc.
* `ts_mod_1week`: timestamp modulus 1 week. We similarly hope this will reveal weekly patterns (are “mondays” a bad day? etc.)
* `attempt_num`, `attempts_correct`, `attempts_correct_avg`: about 11% of the questions were **repeated** questions, so it made a lot of sense to keep a record of which question had been answered by whom and how many times it was answered correctly. This was revealed by @aravindpadman in his great thread: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/194266 and it was an extremely demanding feature to code since it takes a total of 11.2 GB of space at inference all by itself.

# Attention

Our architecture follows the auto-regressive application of sequence to sequence transformers, so a causal attention mask is needed to make sure a given interaction cannot attend to interactions in the future; however there is an exception to this which we believe is critical: interaction grouped by the same task_container_id.

We compute separate attention masks for the encoder and decoder, preventing the encoder self-attention from attending past interactions if they belong to the same task_container_id, and conversely modifying the decoder self-attention to allow to attend to all interaction belonging to the same task_container_id, we further restrain the output of the encoder to the decoder with the encoder attention to prevent a leakage of information from the residual connections in the encoder.

```
causal_mask  = ~torch.tril(torch.ones(1,sl, sl,dtype=torch.bool,device=x_cat.device)).expand(b,-1,-1)
x_tci   = x_cat[...,Cats.task_container_id]
x_tci_s = torch.zeros_like(x_tci)
x_tci_s[...,1:] = x_tci[...,:-1]
enc_container_aware_mask =  (x_tci.unsqueeze(-1) == x_tci_s.unsqueeze(-1).permute(0,2,1)) | causal_mask
dec_container_aware_mask = ~(x_tci.unsqueeze(-1) == x_tci.unsqueeze(-1).permute(0,2,1))   & causal_mask
```

# The Blindfolded Gunslinger

We made a joke in the [meme thread](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208356#1137339) about us wanting to ensemble multiple models, but the competition having only 9 hours to run full inference...

It was already reported that the public test set was sitting on [the first 20% of the test set](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/203383#1115680) so it is possible to maximize the allotted time for the private test set by skipping model inference in the first 20% and predicting only the last 80%. 

We implemented dynamic ensembling that attempts to perform as much ensembling as it is possible in the allotted time for the last 80%.

We dubbed this idea “The Blindfolded Gunslinger” because it fires two guns (models) as much as it can (after a while it will only fire one) but it is blindfolded in the sense that the public LB will be ~0.5 so we cannot be sure if it worked or not until now…

# Hardware

* 1 computer with Ryzen 3950x (16c32t) + 64 Gb RAM + 1x3090
* 1 computer with Threadripper 1950x (16c32t) + 256 Gb RAM + 6x3090

We set up the big computer during the competition which was a project on its own:



Also in the last 8 hours of the competition we rented a 190 Gb RAM + 6x3090 but it did not help us much.

# Software

We used pytorch 1.7.1, fastai and we trained using distributed training and mixed precision (both as implemented in fastai). 

For inference we included the last 500 interactions and summaries in both pickle files and memory-mapped numpy matrices.

# Source code

Available at: https://github.com/jamarju/riiid-acp-pub
