# 4th Place Solution

Competition: nfl-health-and-safety-helmet-assignment
Rank: #4
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285007

# 4th Place Solution

Congrats to winners and thanks to @robikscube for hosting it. It was really a fun competition. We took [this](https://www.kaggle.com/robikscube/helper-code-helmet-mapping-deepsort) as the starter notebook and improved until it gets the 4th place.

I will explain the solution pipeline step by step:

## Helmet Detection

* I replaced the baseline helmets with a trained yolov5 model.
* 15 epochs training with default parameters on the provided extra images with image size 640.
* 5 epochs finetuning on the every 3rd frame of video images with image size 1280 excluding V00 and H00 boxes.
* Detection on test set videos with image size 1280, confidence threshold 0.03 and iou threshold 0.2.

Training our own model was better than using the baseline helmets since we could use the video images for training and have control over the detection parameters.

## Deepsort

* Tuned parameters of the deepsort config.
* I updated the deepsort code so that it returns unconfirmed boxes too.
* I replaced the merge_asof logic in the starter notebook with greedy minimum distance assignment to map tracks to helmets.
* Helmets that couldn't be assigned to any track became a cluster of one helmet.
* Helmet track confidence is calculated as sum of helmet detection confidences within the same cluster. It is used for selecting top ~22 helmets in 2D matching step.

In the last days, I replaced the deepsort with my custom tracking. It scored significantly worse on Public LB (I had no time and hope for testing its validation performance), therefore we ignored it but it turned out to be our best Private LB submission. In summary, it was using helmet similarity score and iou score together to track helmets rather than thresholded step by step approach. And its feature extractor model was trained within ArcFace Team Detection model.

## 2D Mapping

I skip this part for now. @rytisva88 can explain this part better since he worked on it. 

## Jersey Number Prediction

[Jersey Number Training Data]

While jersey numbers are not always visible, we can still utilize them when they are clearly visible. With cropping the area around the helmet, I have created a jersey number dataset. I trained a 2 head model (one head for each digit) with resnet34 backbone. Despite resnet34 is small, it was still very easy to overfit on this data in just 2 epochs. Therefore some augmentations were needed:
* Cutmix with 40% probability where center of the image and the border of the image are from different images.
* Put random 2 digits with Times font on the center of the image and change the target accordingly with 20% probability.
* Convert to black and white with 50% probability.
* Invert colors with 50% probability.

Given the noise, this model didn't have high recall but it has high precision. And having only ~22 players (jersey numbers) available helped the model to eliminate impossible jersey numbers. When the model is confident, labels from 2D mapping is overwritten by jersey number model. 

[Early jersey number model results]

## Cluster Ensembling

For each helmet h in each frame f, I calculate score for each label prediction it has within its cluster c: `np.log1p(conf[c][label])*conf[c][label]/conf[c].sum()`
All helmet-label possible matches within a frame is sorted according to this score. Then they are assigned one by one starting from the highest score and same label is not assigned twice.

## Linear Regression

After cluster ensembling, we may have some helmets not assigned to any label. For covering those, I fit a Linear Regression model within each frame left and top of helmets being the target and x, y of tracking data being the points. So model does the mapping between these 2 2D spaces.

I create a distance matrix with model estimations and helmet left, top values. If we directly use this distance matrix, it performs very close to just doing 2D mapping. Therefore, for each (label, helmet) pair from cluster ensembling, I subtract 100*match_score to bias the matrix towards cluster ensembling output. Then greedy minimum distance assignment is done and all helmets get a label.
