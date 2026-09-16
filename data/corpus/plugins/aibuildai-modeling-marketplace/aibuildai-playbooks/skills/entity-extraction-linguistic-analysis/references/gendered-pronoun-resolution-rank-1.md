# 1st place solution summary

Competition: gendered-pronoun-resolution
Rank: #1
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90392#latest-544753

Great competition and thanks to all the participants for keeping the competition alive!

I was more active initially but got sidetracked by things at work and international travel during the later half.

Since I'm traveling, I'll be able to post only a brief summary now but hopefully update it with more details later.

#Summary

The focus of my work was on 1 architectural contribution and a lot of generalization analysis.

Here are the key ingredients-

1. Data

   - The automated pipeline is illustrated below.
    [Data pipeline]
   - Data augmentation for neither instances - the under-represented category.


2. Architecture

   - Bert finetuning - no hyper parameter tuning was done.
   - **Evidence gathering by Distant Supervision** - various attention-pooling, self-attention, co-attention and evidence decomposition layers to incorporate noisy information from heuristic and pre-trained coref models. Lee et al is a very powerful coref model and grossly undermined by the GAP paper. Here's [https://www.kaggle.com/sattree/3-a-better-baseline](https://www.kaggle.com/sattree/3-a-better-baseline) a kernel that compares various coref models. It established a better 'unsupervised' baseline than the official bert baseline of .62. Unfortunately, I couldn't make it work on kaggle kernels and by the time I gave up (and subsequently got sidetracked) a better bert baseline of .53 had been established.
   - Label smoothing - to handle noisy labels.


3. Unbiased model averaging

   - Averaging across folds
   - Averaging across 5 seeds
   - Averaging across LMs (cased and uncased) - cased models are more representative and help identify named entities but at the same time rare entities often have sparse distributional properties. On the other hand, uncased models are quite noisy.
   - Averaging Sanitized and Unsanitized label models - since the labels are crowd sourced, it is reasonable to assume that there will be some pattern to the errors humans make and will translate to unseen stage 2 data as well (a common challenge in language datasets). At the same time, some of those errors will be just due to the noise in the behavior and enthusiasm of the turkers.

   The intractability of hyper parameter tuning for deep learning models and risk of underfitting/overfitting to the validation set (early stopping) make averaging more useful than usual. However, this effect can also be achieved through SWA by investing some research time but at a substantially reduced computational budget.

Those are amongst the major contributors that I can think of right now!

##UPDATE

The above description is for submission model 2.
Model 1 has all the same elements other than the last bullet point in model averaging. Model 1 was trained only on the sanitized version of the labels, in case the organizers end up putting in some extra effort of cleaning the crowd sourced labels.

###Architecture details

Based on the cluster predictions (clusters co-referent with A, B and P) from the coref models, a mask corresponding to each cluster is created from word-piece tokens and applied to BERT embeddings (from the last layer). This masked version of embeddings for each such cluster is then fed to the *Evidence Gathering* module which systematically aggregates the information and comes up with an *evidence vector*. The evidence vector is then fed to the decision making layer resulting in a probability distribution over A, B and P. 

The *Evidence Gathering* module sits on top of the BERT pre-trained models, but the BERT layers themselves are not frozen and get weakly adapted through error propagation, while, retaining most of the language characteristics necessary for generalization. 

##UPDATE - 06/04/19

arxiv paper link - [Gendered Ambiguous Pronouns Shared Task: Boosting Model Confidence
by Evidence Pooling](https://arxiv.org/pdf/1906.00839.pdf)
github code link - https://github.com/sattree/gap

[Architecture]
