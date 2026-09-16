# 8th Place Solution: single model with heavy augmentation + tuned threshold

Competition: hubmap-hacking-the-human-vasculature
Rank: #8
Source: https://www.kaggle.com/c/hubmap-hacking-the-human-vasculature/discussion/429352

Thanks to Hubmap for hosting such a nice competition and to the many helpful notebooks and discussions.

- Summary
  - Train using all train data with the yolov8x-seg model with strong augmenation
  - The threshold for converting mask prob to binary seems to be a key for scores
- Training
    - yolov8x-seg model( with settings below) with all train data
    - Trained all 3 classes 
    - 2-fold validation 
        - fold1: wsi1, 3
        - fold2: wsi2, 4
        - The key insight for me here was that the optimal value of the mask threshold varies considerably with fold1, 2. For example, in one experiment, the following score were obtained
          - mask_threshold, fold1 val , fold2 val 
          - 0.2, **0.349**, 0.35
          - 0.5, 0.211, **0.481**
        - Therefore, in final 2 submission, I thought it was a safe bet to choose different thresholds for masks
    - The yolov8 setup is as follows
```yaml
imgsz=512,
batch = 16 * 4

lr0 = 1e-4
lrf = 1e-2
cos_lr=True
optimizer = "AdamW"
close_mozaic = 10

## augmentations
hsv_h= 0.015
hsv_s= 0.7
hsv_v= 0.4
degrees= 45.0
translate= 0.1
scale= 0.5
shear= 15.0
perspective= 0.0
flipud= 0.5
fliplr= 0.5
mosaic= 1.0
mixup= 1.0/3
copy_paste= 1.0/3

mask_ratio=1
```

- Inference
  - image size: 768 (better than 512, why?)
  - as noted above, two different thresholds was used
        - thresh 0.5: private **0.56**, public: 0.391
        - thresh 0.2: private0.369, public: 0.506
- Not work for me
    - combined with semantic segmenation
    - Training on large image sizes (768, 1024)
    - tta(rot90) (a bug in my implementation?)
- Did not try
    - Pseudo labeling in instance segmentation model
    - Stein augmentation
    - WBF
    - And many more models
