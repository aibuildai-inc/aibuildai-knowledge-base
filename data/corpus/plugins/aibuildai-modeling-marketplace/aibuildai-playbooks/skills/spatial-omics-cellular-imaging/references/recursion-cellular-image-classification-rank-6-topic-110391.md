# 6th place solution

Competition: recursion-cellular-image-classification
Rank: #6
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110391

Hi kagglers! Nice one, congratulations to all the winners and all who've learned something new. Congratulations and many thanks my team Andriy ( @ayaroshevskiy ), Sasha ( @pajari ),  Vitaliy ( @ladler0320 ). Short summary of our very simple and straightforward solution.

About "**leak**". I won't argue about the definition. But actually, that was simply target class co-occurance and I recommend to check this in every EDA of every multiclassification challenge same as you check class imbalance. EDA is must.
For example:
`df.groupby(["experiment", "plate"]).sirna.apply(lambda x: x.sort_values().unique())`
So why do they co-occur that way? Why experiments are shuffled only within the plates? I guess its by design hard to shuffle them every single experiment for 1100+ classes but anyway that knowledge reduced an error significantly.

Also, as you already know - crucial was to solve **assignment problem** having raw/softmax predicts from neural network. So we've used hungarian algorithm as well. 

- Validation. I've used 30/33 experiments as a validation splitted by **3 folds**. (3 other experiments went to all the folds). Iteratively, I've manually swapped experiments to align their scores. HUVEC-18 went to the train too. We've found that public LB contains U2OS-04 and it's hard to achieve good score for that particular one. I removed it from most of our submissions to see score for other tree cells. But also that gave us a chance that someone will overfit to public LB. Private U2OS-05 was alright and looked much more like U2OS-03/U2OS-01.

- Image **normalization**. Organizers gave us pixel stats so we've used it in exact way - scaled all experiments by their mean/std. By the end of competition I've found that scaling U2OS-04 with mean/std of U2OS-02 increased its score by 10%. But who cares about public LB experiments ¯\\_(ツ)_/¯

- Networks. We've trained our first classification model with softmax - efficientnet-b0 in kaggle kernels before the gpus quotas.  Then in my 2x1080ti devbox. I didn't have much time for modeling because of lack of gpus but I've found EfficientNets work comparably better than densenets and resnets/se-resnets. One fold of b0 takes about 9 hours but the CPU was a bottleneck.

- Hard **augmentations**. All flips/rotates. Hard Brightness and Contrast with prob=0.8. Gamma, ChannelDropout. One thing I've found - uint8 augmentations work worse than float because of distribution discretization. 

- Training process. I train everything with **Adam**, ReduceOnPlateau and few warmup iterations. I've spent so much time trying to understand why people declare SGD works better but no chance - Adam performed better as usual.

- **Controls**. We've simply added train and test (!) controls as additional classes 1108:1139. That boosted score significantly. 

**First iteration**. Pretrained **EfficientNets** b0, b1, b2 achieve almost the same score but blending their raw predictions and than solving assignment problem boosted score nicely. Averaging this 3 models scores 0.744/0.750 (3 cells, U2OS-04 reduced) public and 0.990 private LB (13th).

I saw few strategies of how to split test experiments for **pseudo-labeling**. And actually I've tried around 6 of them with one or two iterations of PL. By the end I've came with very careful strategy - soft labels and balanced experiments split between folds:
```
pl_all_folds = ["HEPG2-10", "HUVEC-19", "RPE-10", "U2OS-05"]
pl_folds = [["HEPG2-08", "HUVEC-17", "HUVEC-23", "RPE-08"],
        ["HEPG2-09", "HUVEC-20", "HUVEC-21", "RPE-09"],
        ["HEPG2-11", "HUVEC-22", "HUVEC-24", "RPE-11"]]
```
As you see U2OS-04 was reduced.
**Soft** pseudo-labelling seemed to work better than hard. Now I understand that the best strategy might be to add them iteratively based on some threshold after solving assignment problem.

**Second iteration**. Same b0, b1, b2 now with pseudo-labels. People don't usual train same architectures in PL stage so I've added b4 and b6 as well. EfficientNet-b4 scored best validation and LB score and unfortunately EfficientNet-b6 didn't converge well in time. For big models we used **gradient accumulations** up to x12. TTA: 2 sites with 8 combinations of flip/rotate each. 

What didn't work:
- SGD with CLRs (triangles, cosine etc)
- Lookahead and Ranger
- ArcFace (achieved almost the same score as Softmax)
- My favourite se-resnext50
