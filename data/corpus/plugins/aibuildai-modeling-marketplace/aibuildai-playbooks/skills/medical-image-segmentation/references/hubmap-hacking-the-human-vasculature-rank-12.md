# 12th place solution

Competition: hubmap-hacking-the-human-vasculature
Rank: #12
Source: https://www.kaggle.com/c/hubmap-hacking-the-human-vasculature/discussion/428319

I am so happy to have won my first gold medal and even a solo medal in this competition. Also thanks to the  OpenMMLab, SenseTime and The Chinese University of Hong Kong for making a really great library mmdetection.

**Summary**
I used the pseudo labels of Dataset3 to train the Cascade Mask RCNN + Convnext v2 large.

**Model**
- Detector : Cascade Mask RCNN 
- Backbone : Convnext v2 Large
- Loss : FocalLoss

**Augmentation**
- Albumentations
   - Distortion worked very well for me.
```python
            dict(
                type='OneOf',
                transforms=[
                    dict(type='OpticalDistortion', p=0.3),
                    dict(type='GridDistortion', p=0.3),
                    dict(type='ElasticTransform', p=0.1),
                            ], p=0.5),
```

- mmdet augmentation
   - AutoAugment, MixUp, Mosaic, RandomErasing

**Training process**

First, weighted segment fusion was performed on the output values ​​from the 4 models made of 4 types of data sets to create pseudo labels for Dataset3.

I did WSF by referring to the guide below -> https://www.kaggle.com/code/mistag/sartorius-tta-with-weighted-segments-fusion



second. The model was pre-trained using the pseudo label of Dataset 3 and the human label of Dataset 2.

Finally, the model was fine-tuned using only Dataset1 data and submitted.


