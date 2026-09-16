# 22nd place solution: Yolov5l + Deepsort + CPD registration

Competition: nfl-health-and-safety-helmet-assignment
Rank: #22
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285179

Congratulation to all the winners!

My best private score is 0.804 but CV score is not high as my selected notebook.

My solution Pipeline

## 1. Yolov5l
- I trained single model yolov5l on 9k images with full resolution and validate on ~4k images extracted from 10 training videos
- My submission is the ensemble of 3 checkpoints that have the highest mAP.
- I also try Yolov5l6 and yolov5x, but training time is too long so I give up :))

## 2. Deepsort
- I extracted features for each helmet box from Deepsort feature extractor using the pretrained weight (ckpt.t7) and save it to disk for step 4

## 3. Point set registration
- I simply use CPD non-rigid registration from https://github.com/neka-nat/probreg to transform box coordinates to tracking point plane
- For each image, I rotate -45 and +45 with step of 10 to find the transformation that has the lowest chamfer distance
-  Then I assign labels to each helmet by using the Hungarian algorithm, the cost matrix is the distance between each helmet coordinate and tracking points

## 4. Post-processing
- For each helmet bounding box, I collect all bounding boxes from previous/next frames that have the same deepsort cluster and assign score to each label by multiplying box confidence and cosine similarity of helmet features (from step 2) as following

```
cosine = np.dot(feats, feats.T).T[0,1:]
score = row.conf.dot(cosine)
```

and then re-assign label that has the highest score.
