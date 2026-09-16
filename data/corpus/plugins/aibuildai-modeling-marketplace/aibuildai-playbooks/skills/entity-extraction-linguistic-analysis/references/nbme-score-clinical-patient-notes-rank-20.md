# 20th Place Solution

Competition: nbme-score-clinical-patient-notes
Rank: #20
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323094

Thanks a lot everybody for the competition, and especially to my teammates @cdeotte @horsek @shinomoriaoshi for the great experience :)

### Pseudo Labels

Our solution (like most of the winning ones) heavily relies on pseudo-labelling. 
The pipeline described below summarizes it, and we report improvements on CV and LB for a single deberta-(v3)-large. 

<a href="https://ibb.co/85fj3r3">[PL-drawio-2]</a>

- Additional details :
 - Pseudo labels were generated on a blend of models (usually 2 or 3 models)
 - Some of our models used 3 iterations of PLs, some used 2, some used 1
 - Models were trained with different proportions of the PL data varying from 20% to 100%
 - We had different pipelines and used different sets of pseudo-labels to keep some diversity. 
 - Some of the models used soft PLs

In order to squeeze a bit more from our pipeline, some of the models were trained with pseudo labels averaged for the 5 folds. This made CV useless but resulted in stronger PLs and a consistent +0.001 public LB boost for our single models.

### Model diversity

- Backbones : 
 - deberta-v3-large, deberta-large, deberta-xlarge, electra-large, roberta-large
- Heads : 
 - LSTM head, 2-layer CNN head on the last or 8 last layers, 1 layer fc head on the last or 8 last layers
- Some models were MLM-pretrained
- Some models used data augmentations (cf comments)

### Post-processing

Nakama’s baseline had issues with start characters, and you could get a decent (0.001-0.002) boost simply by fixing that  (cf comments). In the end we just used Theo’s Roberta notebook to convert token probabilities to character spans. We couldn’t find a more reliable post-processing.

### Ensembles

The CV of ensembles did not correlate well with LB, probably because of label noise, so we just decided to blend together all of our strongest models. Scores are public/private LB.

- Tri (weight=0.2)
 - deberta-large : LB 0.890/0.890
 - electra-Large : LB 0.891/0.889

- Theo (weight=0.6)
 - deberta-v3-large : LB 0.892/0.892
 - deberta-large : LB 0.892/0.892
 - deberta-xlarge (2 folds)  : LB 0.891/0.892
 - roberta-large : not submitted but about ~0.891/0.891
 - deberta-xlarge (2 folds) : not submitted but ~891/0.892

- Chris (weight=0.1)
 - deberta-xlarge (2 folds) : LB 0.891/0.892

- Hiro (weight=0.1)
 - deberta-v3-large not submitted

We had 30 submissions scoring better than the one we chose, including single models in the gold range. But the difference between a mid-range 0.892 sub and a 0.893 sub is actually really small so there’s not really anything we could do about it. 


### Final wise words by @shinomoriaoshi 

> Even though we got the shake-down and I missed the chance to go to the Master tier again, the learning opportunity is much more valuable.




*Thanks for reading !*
