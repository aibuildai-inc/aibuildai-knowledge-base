# 2nd Place Solution: Helmet Detection + 3d Impact Detection + Post Processing

Competition: nfl-impact-detection
Rank: #2
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/208979

Thanks to both the organisers of this competition and to all of the other competitors - well done to everyone who worked hard over the holiday period!

@ilu000 and @robikscube are now competition GMs! Well deserved - congratulations!

Here's the solution from the team @christofhenkel, @ilu000, @philippsinger, @robikscube, @fergusoci


# Brief Summary

Our solution is a two stage approach. Our first stage is a YOLOv5 model which detects helmets in each frame of the video. The Stage 2 ensemble consists of various 3D CNN models which use cropped helmet locations and +/- 4 images from frames surrounding the cropped location. Our stage 2 models predict targets by impact type. Finally, we apply post processing to our ensembled staged 2 predictions.



# Detailed Summary

Below we give a detailed description of our solution:

## Validation strategy

We used GroupKfold by GameID to create a 5Fold split of the data.

We trained various stage 1 models for helmet detection, both on our KFold splits and on the entire training set. We determined the maximum possible score of a stage 2 model based on our stage 2 bounding boxes would be 0.9868 (out of 1888 impacts, the stage one model predicted matching boxes for 1863 of those).

## Pipeline

We approached the competition collaboratively, with each team member working off one central pipeline. We used the following:

- Github: Versioning and code sharing
- Neptune: logging and visualisation
- Kaggle API: dataset upload/ download
- AWS: data storage
- Push kaggle dataset github action: https://github.com/marketplace/actions/push-kaggle-dataset For automatically pushing github commits to kaggle.

We are planning to release our code soon after some clean up.

## Architectures

Our best stage 2 ensemble consisted of 6 different efficientnets + horizontal flip tta. We replaced the first 2d conv layers in the inverted residual blocks of efficientnet with 3d conv layers as @davletag did in the [Deepfake competition](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/158158). Hat-tip @davletag, that also seemed to work well here.

At the last minute we trained 3D Resnext50 & 3D Resnext101 (pretrained on kinetics dataset from https://github.com/okankop/Efficient-3DCNNs). Although initial results looked promising we lacked time to properly tune hyper-parameters and did not use it in our selected submissions.

## Training strategy/ schedule

### Stage 1: Helmet detection

During the competition we trained YOLOv5 per fold using our GroupKfold splits but also adding the extra images. Fortunately, YOLO works out-of-the box with DDP, by simply running torch.distributed.launch so we could speed up training a lot. We used the standard config with pretrained weights and applied put flags

--batch-size 128 
--img-size 704 
--epochs 50 
--data dataset_fold0.yaml 
--weights yolov5x.pt 
--single-cls 



For our final submission, we trained on the full data without a validation set. For inference we saw that using the built-in TTA flag and using an img-size of 1280 gives nice improvements of box predictions. We kept all boxes, which had a confidence level greater than 0.05 or 0.1 and went on with assigning labels. Since our stage1 model only detected helmets, i.e. is a single class detector, we need to assign impact/ non-impact labels after we generated the boxes. For that we took the GT boxes and assigned an impact to each predicted box that had an overlap of 0.35 IOU with an impact GT box. 


### Stage 2: Impact detection from helmets

Input data for our 3D models was a stack of 9 helmet detections from Stage 1: the central crop +- 4 frames on either side. The detection was padded so that we cropped 2x width and height of the original. Some of our models included the original box size as context.

We used the different impact types for the centre frame as our output variable (no impact, helmet, shoulder, body, ground impact) and optimized a softmax loss with class weights split 0.8:0.2 (non-impact : impact). We had approximately 1.1M detections from Stage 1 across all training videos, so we used the following sampling scheme during training:
 
At each epoch we selected all of the positive impact samples and a random sample of negative impact samples according to a specified ratio (we used 0.99:0.01 non-impact:impact). The negative impacts were sampled without replacement so that we saw as many diverse images as possible over the course of training. 

Augmentations included: HorizontalFlip, CoarseDropout, GridMask, ShiftScaleRotate, RandomBrightnessContrast, Blur.

We trained 3D models with efficientnet-b3 and b5 backbones (2d conv layers converted to 3d as outlined previously), and various heads. Each model was trained with a cosine annealing scheduler and Adam optimizer for 30 epochs. Pytorch’s distributed data parallel (DDP) and saving a cache of 9x3 channel boxes to disk allowed us to speed up the training process significantly.



## Post-processing

As the metric was a bit special, and we did not find a good way to reflect it directly within a loss function, we had to rely on certain post-processing consisting of several steps outlined below:

- Filtering stage 1 boxes: For inference, we remove all boxes predicted by our stage 1 model that exhibit a confidence score below ~0.4. This not only speeds up our inference, but also quite significantly improves our stage 2 predictions. Note that removing those boxes also from training stage 2 does not have much impact.

- Filtering early frames: We additionally filter out any frame earlier than 25 as there are usually no impacts detected, this has very minimal influence as our models only very, very rarely would predict impacts in those frames but further speeds up inference.

- Video NMS: This was one of our most important aspects of PP, specifically with weaker models a bit earlier in the competition. The idea was to apply NMS with IOU threshold of 0.5 to all frames of a single video combined. This means that NMS filters out duplicate boxes in subsequent frames and only keeps the ones with highest stage 2 scores. This benefits the metric quite well, as it removes quite a lot of FPs. It got a little bit less important in our latest models that on their own were stronger, but is still an integral part of our pipeline.

- TOPK filter: For each video, we only consider the top 19 predicted boxes based on their stage 2 score. This matches the average number of impacts in training videos and appeared to be a very solid solution to balance TPs, FPs, and FNs. 

- Final threshold: Additionally after topk filtering we remove any leftover prediction below a threshold of 0.15.

We evaluated the parameter of these PP steps on CV, and they turned out to be very robust on private LB and also public LB (even though it was quite tiny).

## Blending

Our original idea was to blend stage 2 models on different stage 1 models leading to different bounding boxes that makes blending not trivial. To that end, we experimented with WBF to blend the different boxes and assign scores to them. However, this not only extended our inference runtime due to multiple stage 1 models, but also did not lead to satisfying results. Ultimately, we settled on a simpler approach that only uses one single stage 1 model (fitted on full data) and then simply averages the scores of all stage 2 models on the same boxes.  




# What did not work

- Determining teams’ helmets using KMeans clustering of RGB colors and adding as features.
- Applying Video NMS to each team individually (based on the team clusters above).
- Adding bounding boxes for all helmets as additional channels before feeding into stage 2 model.
- Adding additional context such as: frame number, view, x/y coordinates of bbox center.
- Adding more or less than +/-4 frames padding, or applying gaussian smoothed weights to the target.
- Training on padded weights.
- Dual-View siamese stage 2.
- Syncing tracking data with the video - and then adding frame level features like player speed, direction, and relative distances, etc.
- Different stage 1 models. None outperformed yolov5.
- Imputing or removing stage 1 bounding boxes based on boxes in previous/post frames.
- Attempting to track and label individual players throughout the entirety of the video.
- Using SORT for improving stage1 boxes
- CenterTrack as a single stage model
- Learning coordinate transformation between views
 
Thanks for reading. Questions welcome.
