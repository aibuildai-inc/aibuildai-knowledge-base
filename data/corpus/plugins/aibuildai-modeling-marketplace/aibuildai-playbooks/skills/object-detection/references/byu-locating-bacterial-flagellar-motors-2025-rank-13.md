# 13th Place Solution - 2.5D YOLO Ensemble with DBSCAN

Competition: byu-locating-bacterial-flagellar-motors-2025
Rank: #13
Source: https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583164

**First of all, I would like to thank @andrewjdarley and @fautei for their great notebooks which were very useful, and @yyyy0201 for sharing valuable insights and ideas.**

I’m happy to finish in the top 50 for the first competition in which I invested time.

My solution involves ensembling 6 2.5D YOLO models with DBSCAN. I’ll describe it in four parts: labeling, preprocessing, training, and postprocessing.

It was quite hard to train multiple models and regularly compute CV scores of my pipeline as I only had access to Kaggle's computing resources.

## Labeling

I used tomograms containing one or more motors. I also corrected some mislabeled data by running my inference pipeline on the training set, followed by visual inspection. I didn't use random slices from tomograms without motors as they would mostly be useless noise. However, using the hard negative slices that my models struggled with could have improved the results.

The external data from @brendanartley was resized to a low resolution and I didn’t have the technical resources to load the original tomograms and apply the desired preprocessing. Therefore, no external data was used, although it could have improved results since YOLO performs better with a large number of training images.

I used 3-4 slices below and above the slice containing the center of the motor along the z-axis. The bounding boxes sizes were 24x24 and 30x30.

I randomly split the data into 80% for training and 20% for validation. After reviewing the slices in both sets, I noticed a good tomogram distribution, thanks to a lucky seed. It helped to get strong solo models with good generalization. I also applied few augmentations to the validation set including gaussian, median, average blurs, CLAHE and RandomBrightnessContrast.

## Preprocessing

The preprocessing only consisted of 2nd and 98th percentile normalization. During inference, the slices were resized to 1024×1024 using letterbox.

I decided to join this competition to focus exclusively on 2.5D models. During inference, the inputs given to the YOLO models were RGB slices with slice z-2 in the R channel and slice z+2 in the B channel. During training, slice z-1 and z+1 were used in R and B channels. The LB scores were better using 2 slices below and above.

## Training

My final pipeline ensembles 6 YOLO models: 8s, 9s, 10m, 2x 11s, 11m.
The training args:
```python
- epochs: 50
- batch: 8 - 16
- imgsz: 960
- dropout: 0.1
- lr0: 0.0001 - 0.0005
- lrf: 0.1
- weight_decay: 0.0005 - 0.001
- scale: 0.4
- mixup: 0.1 - 0.2
- copy_paste: 0.0
- mosaic: 1
```

I also modified the default YOLO augmentations. They are the same as those I applied to the validation set.

I noticed that high-resolution tomograms didn’t contain any motors but as I said in **Preprocessing**, I didn't use hard negative. My models were trained on tomograms with resolutions ranging approximately from 920 to 1000.

My best solo models (8s, 11s) scored both 0.83+ (it could still be improved).

## Postprocessing
 
For each model, the confidence threshold was set to 0.35 and the top 10 detections were kept.

I used TTA (h-flip, v-flip) during inference but I didn't merge the TTA detections using NMS or WBF. I decided to ensemble the detections using DBSCAN. I normalized the 3D coordinates of the detections. Selecting the `eps` parameter was kinda easy, I set it to 0.02. Selecting the `min_samples` parameter was more challenging. Based on my results on the public LB and in order to avoid missing true positives on the private LB, I set it to 22. 

I noticed that clusters elongated along the x or y axis were more likely to be false positives, whereas valid detections were typically elongated along the z-axis but I didn’t follow up on that idea.

During inference 3 models were running on even slices and 3 models on odd slices.

It was an interesting challenge thanks to domain shift and I learned a lot.

Thanks for reading,
Victor
