# 6th place solution - Ultralytics YOLO

Competition: byu-locating-bacterial-flagellar-motors-2025
Rank: #6
Source: https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/587410

Thanks to kaggle and everyone. I start by host's sample code, and train several Ultralytics' YOLO models by different configs.

# Summary
- Ultralytics YOLO models
- Apply filter for denoise
- More data augmentation
- Thresholding strategy
- External data by @brendanartley 

# Data process
- Use [z-3, z, z+3] slices as RGB channels input.
- Use hamming window to filter data along the z-axis to perform denoise.


# Augmentation
Modify ultralytics/data/base.py for more augmentation, like gamma and random size.

# Thresholding
Two strategy, both perform well:
- Thresholding by 56 percentile
- Auto Threshold：
`
s = np.sort(confidence_score)
`
`
r = [ ( s[i-50] + s[i+50] - 2*s[i] ) for i in range(50, len(s)-200) ]
`
`
confidence_threshold = s[ np.argmax(r) + 50 ]
`

# Ensemble
The private LB score is ensemble by 4 models:


Thanks again.
