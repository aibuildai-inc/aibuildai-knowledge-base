# 3rd place solution

Competition: MABe-mouse-behavior-detection
Rank: #3
Source: https://www.kaggle.com/c/MABe-mouse-behavior-detection/writeups/3rd-place-solution

Here is a link to my final script:

https://www.kaggle.com/code/ymatioun/mabe-blend-ym

It works for both training and submission, based on settings in “nn_mode” and “CFG.mode”. But the training was done offline, with pre-trained models used for inference – for runtime reduction and simplicity.


My solution is a blend of neural network and XGBoost. Several versions of NN and XGB were trained with slightly different settings and blended together.


Neural network is LSTM with several dense layers. Output is “softmax” for all allowed actions. Not allowed actions are masked out. “Laboratory code” runs through categorical embedding layer and is added to the inputs.

Most effort went into feature engineering. I ended up with around 60 features – mostly distances, angles, speeds, etc. Some metadata features.

Input data is combined for all labs, selecting 5 body parts present in most of the data. This way the model is able to learn data patterns in all the labs.

Training was done with small batches (batch size of16 or 32 ), using “CosineDecayRestarts” schedule.


XGBoost model is based on the best public notebook, with many changes. Here are some of them:
    • added new features
    • removed features with low importance
    • early stopping based on F1 metric
    • smooth predicted probabilities before using them
    • optimize model hyper-parameters


Post processing: blended predicted probabilities turned into binary prediction when probability exceed a threshold. Threshold varies by lab and action; it was optimized by grid search on out-of-fold predictions. When multiple actions are predicted for the same frame, the one with the highest ratio of probability to threshold is kept.


Exploiting data properties: I looked into it, but could not get much benefit from it. But there are some potentially exploitable data properties out there:
    • for some lab “AdaptableSnail” videos, actions have been mislabeled, and organizers did not fix this issue – this was discussed in https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/discussion/612531 
    • for lab “BoisterousParrot”, most actions have length as a multiple of 15 frames
    • for lab “CautiousGiraffe”, many actions have length of 9 frames, with mouse 1 and mouse 2 alternating attacks
I suspect that these (and possibly other) data properties could lead to “magic” solutions exploiting them, as happened in the past in many other competitions – will see if that happened here (I’m writing this before looking at other solution write-ups, so don’t have the answer to this question yet).
