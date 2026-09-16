# 🏅 Best single models 🏅

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #25
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/161014

I wonder what were the best singles models around.

From the write ups it seems like most people got XLM-R around 0.942* and heavily rely on ensembling, post-processing and other tricks (not diminishing their amazing achievement).

Anyway, I'd like to share a [single model](https://www.kaggle.com/hmendonca/jigsaw20-xlm-r-lb0-9487-singel-model) (weights and code) with pseudo labels and knowledge distillation that got to **LB 0.9475** (and **0.9487** after applying the hardcoded multipliers from @christofhenkel thx!) here https://www.kaggle.com/hmendonca/jigsaw20-xlm-r-lb0-9487-singel-model

Unfortunately we failed to ensemble as well as others and did not explore the language modifiers beforehand. This comp was an amazing learning experience for me doing NLP for the first time in my life :D

The main take out for me was that relying to heavily on pseudo labels seem to have made blending a much harder task (knowledge distillation seem to have worsen that issue). It took me too long to realize that but lesson learned for the next comps. I'll extend my report when I find some more time, but please let me know your thoughts on that! 
(and please share your models if you can)

Cheers!
