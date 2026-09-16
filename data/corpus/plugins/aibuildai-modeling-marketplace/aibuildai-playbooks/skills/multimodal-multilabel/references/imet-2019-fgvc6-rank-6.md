# 6th place simple solution with code

Competition: imet-2019-fgvc6
Rank: #6
Source: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/95282#latest-550969

Congratulations to all the teams got the medal.
Compared to other teams, my solution is quite simple (though I tried other tricks, none of them worked).

## Solution
My code is available at https://github.com/YU1ut/imet-6th-soltuion.
My solution is based on this kernel https://www.kaggle.com/lopuhin/imet-2019-submission, and I modified it as follows.

1. Change the input size to 320 by ```RandomCrop(320, pad_if_needed=True)```.
2. Add Mixup and RandomErasing during training.
3. Add pre-trained models from https://github.com/Cadene/pretrained-models.pytorch.
4. Make a 40-fold CV and use some of them to train models.
5. Train se_resnext101 by 10 fold (1~10), inceptionresnetv2 by 5 folds (6~10) and pnasnet5large by 5 folds (1~5). As a result, 20 models are trained.
6. Use the average output of all models (#TTA 2). And adjust the threshold for each image according to the max probability of that image. Supposing we have a probability matrix whose shape is (N\_SAMPLES x N\_CLASSES), the binary results can be calculated by 
```
prob_mask = []
for prob in probabilities:
        prob_mask.append(prob &gt; prob.max()/7)
```
This threshold calculation can boost the LB score about 0.005.

## What didn't work for me:
1. Graph Convolutional Networks https://arxiv.org/abs/1904.03582
2. Soft Label and Pseudo Labeling
3. EfficientNet

Thanks.
