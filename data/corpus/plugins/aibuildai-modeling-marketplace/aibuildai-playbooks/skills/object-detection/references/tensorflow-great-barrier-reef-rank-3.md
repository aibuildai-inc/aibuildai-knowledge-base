# 3rd place solution - Team Hydrogen

Competition: tensorflow-great-barrier-reef
Rank: #3
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307707

Thanks to hosts and all participants for this interesting competition. This is a joint writeup with  @ybabakhin (big, big thanks for the awesome team work in this competition) from Team Hydrogen.

#### Summary
Our solution is a blend of five different object detection model types with tracking.

#### Cross-Validation
For cross-validation we used two strategies: one is simple split by video, and the other is making 5-folds based on subsequences. We were mostly looking at subsequence Cross-Validation scores, time-to-time checking the video folds performance. This strategy gave us a good correlation between CV and Public LB scores throughout the competition, however it failed for the Private LB. You can read our thoughts about such a behavior below in this post.

#### Models
Our final solution is a blend of five different model types. All our models are trained on a resolution of 1152x2048 and only using images with boxes. For augmentations we employ a mix of random flips, scale, shift, rotate, brightness, contrats, cutout, cutmix.

An additional augmentation we utilize in most of our models is what we call “background-mix”. Here, we randomly mix an image with boxes, with a random image without boxes. This guarantees us to not overlap / distort any boxes, but generalizes better to unseen backgrounds and also randomizes the intensity of objects. Also some models benefit from using cut&paste augmentation by simply cutting and pasting objects to other frames.

**CenterNet**
We took an idea from the original CenterNet and substituted backbone with HRNet. The object detection task in CenterNet architecture is treated as a keypoint estimation, so we used x4 downsampled labels (288x512) for predicting heatmaps of: objects centers, width and height, and center offsets. Final submission includes two backbones: hrnet_w18 and hrnet_w32 that produced similar local performance. The models were trained for 10 epochs and were using cut&paste augmentation.

**FasterRCNN**
We implemented FasterRCNN to work with any timm backbone and found that NFNet models work particularly well due to generalizability and also easiness to train on small batch sizes. Our final models use eca_nfnet_l1 and dm_nfnet_f0 backbones. We use 4 feature layers and 256 FPN out channels. Models were trained for 10 epochs.

**FCOS**
Similar to FasterRCNN, we integrated FCOS into our pipeline to work with any timm backbone and use the same backbones for training. Here, we use 3 feature layers and also 256 FPN out channels. Again, models are trained for 10 epochs.

**EfficientDet**
We train EfficientDet models using efficientdetv2_dt backbone which has a great balance between complexity, memory usage, and runtime. Models are trained for 20 epochs.

**YoloV5**
We trained yolov5l-6 using rectangular training. We adapted the original code to directly optimize F2 score, and also for rectangular training to properly work with random shuffling, mixup, cutout and other granularities. Models were trained using Adam and for 40 epochs. Inference is done on training resolution without tta.

#### Blending & Tracking
For blending, we employ average WBF blending which is particularly useful here as it automatically incorporates a voting mechanism of boxes to downweight boxes that are not present in all models. 

For tracking, the issue with adding missing boxes using Kalman Filter is quite tricky, as also there it is tough to estimate width and height, and the public kernel solutions do not work well when models get better. So we use tracking in a slightly different way. We use euclidean distance on the center points to find the tracks, but we keep low confidence boxes for finding the tracks. Then, we are setting the confidence of low confidence boxes to the maximum of the confidences of the track. So let’s assume we have a track with confidences 0.8, 0.7, 0.9, 0.15 - in this case we increase the confidence of the 0.15 box to 0.9. The models are quite good in finding the boxes, but sometimes they are just not confident enough, so this is clearly better than estimating the new boxes from the track with Kalman filter or similar estimates. This tracking brings about 0.01 locally.

Another tracking technique that we explored in the last days of the competition was Optical Flow estimation. We used the OpenCV implementation of goodFeaturesToTrack() and calcOpticalFlowPyrLK() methods. It produced pretty nice estimates of diver/camera movements that allowed to better predict the position of starfish across the frames. Unfortunately, it gave only 0.001 improvement in both CV and LB.

#### Label tightness & Public LB
As probably for most people, our public LB was significantly lower than our CV for the first half of this competition. However, we quite quickly noticed that there has to be something different in public LB vs. training data, mostly because people with higher LB has so much worse CV reported than what we saw locally. After a while, the discussion about higher resolution came up, and we just blindly tried it on LB and it immediately pushed individual CenterNet and RCNN models above 0.70. 

So then some funny theories emerged on the forums and obviously we also tried to understand how something like this could happen. After several failed theories, we finally found that higher resolution than training, produces boxes that are more tight (exact) around the COTS. In training, labels are very imprecise, and also inconsistent across frames, it even looked semi-automatic. And the size of the boxes plays a huge role in this evaluation metric setting, as 0.8 IOU threshold has a particularly large impact.

To check this theory, we just submitted manual pixel adjustment on public LB, i.e. decreasing the size of boxes by fixed pixels, and after tuning this a bit, it even worked better than the upscaling, and was much faster obviously. So our theory emerged that test data is labeled better after collecting the following additional evidence:

* Higher resolution inference produces tighter boxes, it works better on Public LB
* Manual pixel adjustment (making boxes smaller) works better on Public LB
* None of these adjustments works on ANY part in training data
* Visually and manually checking a few training examples, we found that boxes on training are on average 3px too large in all dimensions. This is the EXACT same amount that worked well for us on public LB. 
* Hand-labeling a small portion of train, finetuning on it, boosted LB significantly.
* The public LB had sequences from both videos. We checked if the manual adjustment works on both videos, and it did, further strengthening our suspicion that whole test is different.

So we were very confident that it should hold for whole test, but as we know, it did not. So either we completely misinterpreted some of this evidence, or private test is again labeled differently. In general, we personally feel like that public LB should better represent private LB, and in best case it should be similar to training data. This would lead for competitors to overall focus more on improving models, vs. trying to understand the splits.
