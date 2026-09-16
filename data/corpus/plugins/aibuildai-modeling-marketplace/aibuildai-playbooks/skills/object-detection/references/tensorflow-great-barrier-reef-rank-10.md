# 10th place solution — 2xYOLOv5 + tracking

Competition: tensorflow-great-barrier-reef
Rank: #10
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307756

Congratulates all the participants and many thanks to organizers. It was very challenging and interesting competition.

This post was written jointly with [@danjafish](https://www.kaggle.com/danjafish) and we want to say a big thank you to our team.

# Summary
Our solution is an ensemble of two differently trained yolov5 models + tracking.
[summary]

Our final submit code is available here
[https://www.kaggle.com/parapapapam/2xyolov5l6-tracking-lb-0-678-private-0-722](https://www.kaggle.com/parapapapam/2xyolov5l6-tracking-lb-0-678-private-0-722)

# Models
In our solution we used two yolov5 models trained in different ways.

## YOLOv5l6 1280 
(by [@parapapapam](https://www.kaggle.com/parapapapam))

### Idea
The main trick of this competition is the use of high resolution. But when you just resize the image, you increase the noise without adding any additional information, so the reason why it works better not in the image.

My hypothesis was that the reason of that effect is in achors assignment for small object. When you increase image resolution you make your objects bigger, but anchors grid step is the same and you get better allignment so model works better.

Therefore I decided do not increase input resolution, but use higher resolution inside the model. For that I changed stride of second convolutional layer from 2 to 1:

```yaml
...
[-1, 1, Conv, [128, 3, 1]],  # 1-P2/4
...
```

So this model will have 2 times bigger resolution starting from 3rd convolutional layer and should behaves like 2560 model but without resizing the original image.

### Validation Strategy

Split by video. Video 1 was used as a validation. For submit model was retrained on videos 0 1 with validation on video 2.

### Data

I’ve used kaggle data with 6 iterations of fixing the labels. I’ve checked FP errors of models with high confidence and fix them manually using labelImg. It improves CV without any effect on LB. 
I’ve trained on labeled images + 10% unlabeled data picked at random.

### Augmentations

I used albumentations with the following pipeline:

```python
aug_strong = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.OneOf([
        HueSaturationValue(p=0.5, hue_limit=0.05, sat_limit=0.7, val_limit=0.4),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.15, contrast_limit=0.15),
        A.ToGray(p=0.1),
    ], p=0.8),
    A.OneOf([
        ShiftScaleRotate(p=0.4, shift_limit=0.1, rotate_limit=3, scale_limit=(0.5, 2.0), min_bbox=27, max_bbox=None, border_mode=0, value=(114, 114, 114)),
        A.Perspective(p=0.1, scale=0.1, pad_mode=0, pad_val=(114, 114, 114)),
    ], p=0.8),
    A.OneOf([
        A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
        A.MotionBlur(p=0.5, blur_limit=(3, 5)),
        A.MultiplicativeNoise(p=0.5, multiplier=(0.9, 1.1), per_channel=False, elementwise=True),
    ], p=0.2),
]
bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"], min_area=16, min_visibility=0.2)
transform = A.Compose(aug_strong, bbox_params=bbox_params)
```

And from YOLOv5 mosaic=1.0 and mixup=0.5.

Also I’ve used enchancement based on clahe and channel stitching for train and inference. It gave around +0.005 CV/LB.

### Train params

10 epochs, 8 batch size, SGD 0.01, OneCycle, input size 1280x720.

Also I’ve changed obj: 8.0 because it depends on image size and I have to multiply it by factor of 2 ^ 2 = 4 because I’ve used two times bigger resolution inside the model and also experiment shows that multiplying it by 2 also a bit increased CV.

### Results

CV ~0.65 (+ tracking 0.66) / Public 0.67 (+ tracking 0.685) / Private 0.699 (+ tracking 0.718)

### What didn’t work

Copy paste COTS bboxes augmentation. Style transfer. External data.

## YOLOv5l6 3100 
(by [@danjafish](https://www.kaggle.com/danjafish))

My approach was quite straightforward. I got most of my ideas from [https://www.kaggle.com/c/global-wheat-detection/discussion/172569](https://www.kaggle.com/c/global-wheat-detection/discussion/172569) and [https://www.kaggle.com/c/global-wheat-detection/discussion/172418](https://www.kaggle.com/c/global-wheat-detection/discussion/172418). However, I decided to use yolov5 for a start since I found it the most promising.

- Validation strategy: split by video. Video 1 was used as a validation. After selecting thresholds and other parameters, the model was re-trained on all data.
- Data: I used only kaggle data. Trained on not empty images only. Adding empty ones slightly improved scoring on validation, but significantly reduced LB. That's why I gave it up.
- Various augmentations from [albumentations](https://albumentations.ai/) (I borrowed pipeline from [https://www.kaggle.com/c/global-wheat-detection/discussion/172569](https://www.kaggle.com/c/global-wheat-detection/discussion/172569)):
    - HorizontalFlip, ShiftScaleRotate, RandomRotate90
    - RandomBrightnessContrast, HueSaturationValue, RGBShift
    - RandomGamma
    - CLAHE
    - Blur, MotionBlur
    - GaussNoise
    - ImageCompression
    - CoarseDropout
- I also changed the default mosaic implementation to get images of about the same size and made up parts of images with CenterCrop.
- Inference size: 3100. I tried different sizes and different architectures. The best results on validation were shown by yolov5l on size 3000X3000. However it was much worse on LB than l6 3100X3100. That's why I used the later one.
- Train params: 20 epochs, Adam optimizer, lr 0.001 ony cycle sheduler. bs=4
- Hardware: most of the time I used 4 V100 GPUs with 16GB RAM. Model takes about 2 hours to train.
- What did’t work:
    - Manual reannotations of data - some COTS were not labeled because they appeared in frames later than updated annotation. (my assumption). Adding these COTS slightly improved CV, but significantly worsened LB.
    - TTA. Slightly impoved LB and CV, but takes much longer to run.
    - Blend of different yolo architectures
    - Blend one stage and two stage detectors. I spent the last week completely devoted to the models on mmdetection, hoping to improve the ensemble score. However, my cascade rcnn showed much worse results on both CV and LB.
- Final scores CV/Public+tracking/Private+tracking: ~0.643(video 1)/0.693/0.722

# Ensemble

We took predicts of two above models and use WBF with iou_threshold = 0.6 to blend them with weights 0.3 and 0.7. After that we filter results with confidence threshold ≥ 0.1.

But we didn’t spend enough time to tune the weights and the final result seems not much better that each model, but we have strong models, so each of them + tracking already is in gold :).

# Tracking

I’ve already posted tracking approach based on norfair library
[https://www.kaggle.com/parapapapam/yolox-inference-tracking-on-cots-lb-0-539](https://www.kaggle.com/parapapapam/yolox-inference-tracking-on-cots-lb-0-539)
So I will not go into details of norfair tracking implementation based on SORT algorithm.

Here I want to mention that this problem is not a MOT problem because objects are almost static and we have to deal with only camera movement.
For that we’ve developed algorithm of homography calculation between every two consecutive frames using ORB keypoint descriptors. Next we transform our bboxes using homography matrix to obtain predictions on the next frame.It gives much accurate predictions on two consecutive frames comparing to norfair:
[frame1]
[frame2]

Green labels is a ground truth, the blue ones are predictions using just norfair tracking, red ones — homography transform from previous frame + norfair. As we can see, homography + norfair match much better than just norfair.
When we have no detection for object that was detected at least two frames before we predict it 2 more frames using tracking bboxes and if the object doesn’t appear remove it from tracker.
Also we filtered objects that are moving out of the image.

Our tracking approach gives stable +0.01-0.02 CV/Public/Private.

# Other ideas

### Rounding
We found that using of bboxes = bboxes.astype(int) instead of bboxes = bboxes.round().astype(int) improves CV and Public ~ 0.005 and have no effect on Private. It seems the same like was already discussed
[https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307605](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307605).

Thank you for reading and happy Kaggling! :)
