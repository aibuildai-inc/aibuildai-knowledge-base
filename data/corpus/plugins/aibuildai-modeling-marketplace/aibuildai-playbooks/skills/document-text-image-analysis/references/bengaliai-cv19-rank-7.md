# 7th Place Data-based Generalization Solution

Competition: bengaliai-cv19
Rank: #7
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/135960

Many thanks to Kaggle and Bengali.AI for hosting such an interesting competition. It is my first all-in kaggle competition, and needless to say none of this would have been possible without our awesome team Igor, Habib, Rinat, and Youhan. @cateek @drhabib @trytolose @youhanlee 

### Analysis: This competition has a distinctly stratified LB:
- With a good pipeline, single model with baseline augmentations LB ~.97. 
- **Removing crops, training on full-resolution, adding cutmix / cutout, and training enough** should get model up to LB .985+, and with some tweaking ~.989+. These are well-covered in the discussions posts.
- Our best single model is from Rinat, PNASNet-5-Large from [Cadene's repository](https://github.com/Cadene/pretrained-models.pytorch) which scores LB 0.9900, CV .9985. I believe most of the participants LB .9850-.9910 are using ensemble of models scoring around this range.

At this stage (.9900-.9905) we found it very hard to further improve our LB. As [this post](https://www.kaggle.com/c/bengaliai-cv19/discussion/134601) points out, higher CV does not necessary mean better LB anymore, so we were stuck for a while (a long while...).

### Improving Generalization
With the CV &amp; LB relationship it is easy to see that the problem is that our model **could not generalize to unseen graphemes** (by grapheme I mean triplet combination of grapheme root, vowel and consonant diacritics). There are only ~1290 graphemes in training set, while there are 12936 possible combinations. Our models might be biased to output seen triplets; for whatever reason, as Qishen Ha's wonderful [post](https://www.kaggle.com/c/bengaliai-cv19/discussion/134434) pointed out, our models perform badly on these graphemes.

In hindsight, it's no magic that the most convenient way to improve generalization is to **add more data**. When investigating the grapheme representations, I found that **graphemes are actually encoded as sequence of unicode characters**. The roots and diacritics also have their corresponding sequences. The most magical part is that **the unicode sequence for the grapheme is a combination of the sequences of its roots**. Best exemplified using this image, where the four rows are *grapheme, grapheme root, vowel, and conso*.



This observation works (with some minor exceptions) on all graphemes in the training set, and using a very basic algorithm I was able to generate correct labels for all 1295 graphemes in training set except for 2. Following Guanshuo Xu's post pointing to [this repository](https://github.com/MinhasKamal/BengaliDictionary), we are able to access a comprehensive list of grapheme combinations. Using the decomposition algorithm we selected ~3000 graphemes that could be broken down into the given roots and diacritics. Our LB boost on the final day is from better generalization on these graphemes, which partially overlaps with train &amp; test set.

With these graphemes, we rendered them using various fonts and obtained a cleanly-labeled synthetic dataset with ~47K images spanning these characters which can be found [here](https://www.kaggle.com/roguekk007/bengaliai-synthetic-magic), all the ingredients for creating this synthetic dataset has been publicly available. Our hope was that by seeing synthetic images, our models should be able to at least learn the topological (if not stylistic) features of the grapheme. And it turned out they do. Here is a sample of synthetic image in our dataset.



At this point we have only 24 hours until the end of the competition so just finetuned our previous models with the synthetic data added to both original training &amp; validation, thus resulting in the exciting quantum jump on the last day😊 

### Some Details
My teammates have wonderful pipelines, here are some approaches which have turned out useful:
- Train on 128x128 data for ~100 epochs, then finetune on 224x224 data. It definitely yields faster training and maybe better generalization (used in Rinat's pipeline)
- Anneal augmentations except for cutmix gradually to 0 at late stages of training (used in Habib's and Igor's pipeline)
- A wonderful new architecture called MixNet, we found that it nicely complements PNAS5 for ensembling and performs pretty well.
- **Loss:** Plain cross entropy. Tried arface (did not work), focal (did not work), normalized softmax+label smoothing (looked promising but ditched along with my pipeline)
- **Scheduler:** ReduceLRonPleateau for Rinat's pipeline, and hand-monitored dropLR for Habib's and Igor's pipelines.

### Finally
It has always been my dream to do a gold-medal solution write-up before 17th birthday😃 . In the end, more than happy to see hundreds of experiments and weeks of stress translate successfully into solid rank. With more time than 24 hours, we would have: 
1. Searched for a more comprehensive list of graphemes
2. Created more synthetic data with more fonts
3. Somehow make synthetic data more similar to handwritten data; we are using dimming max_val to 230 and gaussian-blur, but can definitely be improved by maybe CycleGAN
4. Make flipping work as pointed out [here](https://www.kaggle.com/c/bengaliai-cv19/discussion/126761). It's really ironic I did not make my own idea work when Qishen Ha did; guess this is the difference🤓 
5. Try more model-regularization methods, ShakeDrop was on the list and looked promising but was preceded by other priorities.

Overall, this has a very exciting and memorable puzzle-solving competition. Again kudos to the team and:
Love Kaggling
