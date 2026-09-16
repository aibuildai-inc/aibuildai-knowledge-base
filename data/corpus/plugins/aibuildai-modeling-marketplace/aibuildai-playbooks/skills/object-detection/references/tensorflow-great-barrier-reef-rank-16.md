# 16th Simple Solution - Only Yolov5

Competition: tensorflow-great-barrier-reef
Rank: #16
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/308336

I would like to thank the organizers for hosting the great competition.
also, I was very honored to become a teammate with @vaillant and @hwigeon. 
I really learned a lot from our teammates. I think it will be the unforgettable good experience.

# Summary
We reached LB 709 so quickly using a large-size inference trick and naive iou-tracker.
but at the same time, we started to doubt the public LB since we couldn't reproduce the result clearly. we found out that Increasing image size increased recall but also yielded unnecessary bounding boxes.
So, we started to build robust models by checking CV F2 scores.
In CV, only inference with originally trained size worked.
We tried size threshold-based prediction, IOU tracker(i might have made a mistake), larger size than originally trained size didn't work in the local.

# Models
We found out that a large model necessarily isn't needed.
Also what we found out is multi-scale ensemble boosts both cv and public LB scores.
Instead of investing our time in searching complex models, we prepared yolov5m trained with the sizes of [2400,2560,2688,2880,3000]
and yolov5s trained with the sizes of [3000,3600,3720,3840,3960,4032,4080,4244]

# Traning Method
We changed yolo's hyperparameters slightly
Applying mixup 0.5, mosaic 1.0, strong hsv change helped a lot
10 epoch training and adam optimizer were chosen.
Model was chosen by weighting Recall vs Precision  4:1

# Validation
We chose video-split folds since these are more realistic.
Ensembled CV scores for video-split were each [0.71, 0.64, 0.77]
After that, we had searched optimized WBF coefficients and confidence scores using a method of grid search.

For video fold 0-1, WBF coefficient 0.5, skip bbox 0.01, confidence score 0.1 was best
For video fold 2, WBF coefficient 0.5, skip_bbox 0.1, confidence score 0.2 was best

With these values, we trained models with all the datasets and regarded the public LB as a holdout
the public LB scores of each model were ranged in [0.55~0.60]

The result is private LB 0.713 and public LB 0.648.
We missed the gold but we couldn't choose the best private LB since this competition is a kind of shake-up competition

Thank you for reading this!
