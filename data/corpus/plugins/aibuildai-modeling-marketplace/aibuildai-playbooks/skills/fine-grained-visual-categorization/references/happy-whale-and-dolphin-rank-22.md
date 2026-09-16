# 25th summary |  Happywhale ?... No, this is DATASET competition 😅😅😅

Competition: happy-whale-and-dolphin
Rank: #22
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/319806

backfin crop,  sod mask/crop,  detic crop ... and many many crop dataset dominate this competition. Thanks @jpbremer releasing their dorsal and full body datasets,  and [MFGA]Make Feature-engineering Great Agein 🤔. We are still in the epoch of "manualy" intelligence .

**about single model**

1. Thanks to backfin crop dataset, we crossed the 0.76 for the first time.
2. Then we aggregated all the manually fullbody annotated datasets,  train yolov5 detector.
3. With the new fullbody crop dataset,  we train backfin/fullbody/sod model,  and combine them, first time cross 0.811,  and then, try larger image size 512 -> 768 -> 1080 ... ,   larger model size convnext-small -> base,   effn-b4 ->  effn-b7,   got some booster on lb/cv,   and some model diversity;
4. Multiple-task training work,  train model on both individual fine-grained task and species classification task give us 0.02 boost.
5. pseudo is efficient, when our ensemble model got 0.851,  we use this submission prediction(without new_individual) train single model,  got 0.850 lb.

**two-stage ensemble strategy**

First,  we use the fold-train models,  predict val dataset,   then calculate the ratio of #1 is new_individual , the ratio of #1 models prediction is same (w/o new_individual)  on each image.  Then do the statistic,  groupby the two ratios and calculate the true label ratio on each group;

Statistic rule apply,  on the test dataset,  we calculate the full-train data #1 is new_individual , the ratio of #1 models prediction is same (w/o new_individual)  on each image too,  and then use the statistic to determine the new_individual place.  The other predictions ensemble by place weight is the same as the public kernel.

**No time to try**

1. transfer learning by species/other similarity image/ original image ...
2. multiple cnn branch model + mask layer.  we have many crop data source ..  backfin/sod/fullbody,  each cnn branch input one,  add some mask layer for combine the cnns output..
3. image embedding ensemble
4. post rank model
5. power of TPU ...

Thanks for sharing, we have learned a lot . 😄 see you in next comp.
