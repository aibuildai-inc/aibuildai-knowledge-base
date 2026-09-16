# [1st place solution] pretrained CNNs -> xgboost -> F1 optimization

Competition: imaterialist-challenge-fashion-2018
Rank: #1
Source: https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57944

Hello Fellow Kagglers!

First of all, I would like to thank competition sponsors and Kaggle for organizing a competition that proved both to be a lot of fun and a very good learning experience!

My solution begins with training multiple architectures. I start off with pretrained models and train them on the training data. I used the [fastai library](https://github.com/fastai/fastai) and [pytorch](https://pytorch.org/) - this is important as both allow me to do things out of the box that as far as I am aware are very hard to do or impossible using other frameworks. 

For training I used Adam initially but I switched to the 1cycle policy with SGD very early on. You can read more about this training regime in a [paper](https://arxiv.org/abs/1803.09820) by Leslie Smith  and you can find details on how to use it by [Sylvain Gugger](https://twitter.com/GuggerSylvain), the author of the implementation in the fastai library [here](https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy).

The difference in results on the train set vs the provided val set raised my suspicion that the data might be coming from different distributions. Upon checking competition rules (which allow training on the val set), I subsequently trained with cross validation on the val set. As training on the test set was not explicitly allowed, I didn't do any form of training on it (for instance, have not tried pseudo labeling).

I grew the pipeline organically and even on the last day of the competition I added a new model (nasnet mobile).

I first attempted to combine model predictions via implementing heuristics for optimizing the f1 micro averaged score. I did a bit of research on this - the paper A Study on Threshold Selection for Multi-label Classification by Rong-En Fan and Chih-Jen Lin is a very nice starting point. It even has descriptions of algorithms one can start implementing! There is not much information available on this on the Internet as far as I could tell so this is quite a nice paper to have in your arsenal.

The optimizing heuristics take 7 minutes to run on a 1080ti. They focus on combining entire submissions / predictions for specific columns and for improving the per column threshold.

I suspected that I could still improve predictions for label k based on combining predictions for all labels. I ran all my model predictions through xgboost with cross validation (5 cv splits, training a single model in each split for the ~170 labels of interest). This entire procedure on an AWS c5.18xlarge instance takes ~ 1hr 15 m. Going forward I am planning on creating docker environments for most of the work that I do so this should make for even easier environment switches and those compute instances come at a really nice price point. Here I think a single round of training cost me ~1.5$ (I use spot instances)

Ultimately, I take raw model predictions, xg predictions based on model outputs, and send it all through my optimization pipeline. 

Below I am providing model results on the validation set. The threshold has been optimized on a per label basis to keep the comparison as fair as possible. I realize that the models have been trained to various degree due to the tools / methods I used, but I think this is still a valuable reference point to have. In particular, resnet 34 and nasnet mobile (just over 5 mln parameters) do stand out as nice goto architectures. I feel they might be great candidates as initial architectures when working on majority of problems and starting with anything bigger could be detrimental to final results. That is of course depending on the resources you have available. I trained on a single 1080ti.

![architecture performance][1]

And here are prediction correlations

![prediction correlations][2]

## What did not work

- darknet50 with attention - I took darknet50 from the [fastai imagenet-fast repository](https://github.com/fastai/imagenet-fast) and added attention. I didn't get far in training the model possibly due to lack of resources (including running out of time that I was willing to spend on this) or an issue with the architecture. I am hoping to revisit this at some point
- per label training - This ended up being quite a time sink of an activity. Very hard won minor improvements, training was very hard to optimize. I am suspecting that the reason why this is so hard might be that the architectures that I used have a lot of capacity and that training on 228 labels takes advantage of multi task learning. 

I am not sure if I am qualified to make such recommendations, but based on the experience of this competition I would like to suggest that if you are considering participating in a Kaggle competition, it might be worthwhile to consider joining one from start. The quality of the experience and what you will be able to learn will be very different depending if you join a competition late or from the very beginning.

See you on these boards and in competitions to come!

Radek

PS. I am very open to discussing anything that might be of interest to you regarding the solution. Also, I generally try to share things I feel might be useful to others. I am hoping to be more active on Kaggle forums but if you'd like you can also connect with me on [Twitter](https://twitter.com/radekosmulski)

  [1]: https://i.imgur.com/LqckkpW.png
  [2]: https://i.imgur.com/ubKtC5Y.png
