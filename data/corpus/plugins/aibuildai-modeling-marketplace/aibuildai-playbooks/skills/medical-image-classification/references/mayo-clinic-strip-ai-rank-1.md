# 1st place solution

Competition: mayo-clinic-strip-ai
Rank: #1
Source: https://www.kaggle.com/c/mayo-clinic-strip-ai/discussion/357892

Thanks to Mayo Clinic and Kaggle for this competition. I enjoyed playing around with it. Also thanks to my teammate @evilpsycho42 for all the support. I will try to illustrate my solution here even though I decided to stop working on it a month ago...

### Data
- Tiling and pick the top 16 darkest tiles

### Single model
- backbone: ```swin_large_patch4_window12_384``` + customized head
- classification head: customized: replace average pooling with attention pooling

### Loss\Metric
- **implement loss\metric following competition metric**

### CV Strategy
- 5-fold Stratified Grouped KFold
    - Stratified by class and grouped by ```patientid```

### What Works
- **attention pooling: 5-fold cv average improves from 0.69->0.66**
- **moco-v3 pretraining: 5-fold cv deviation improves from 0.30->0.15**
- ensemble: 5-fold cv average improves from 0.662->0.658 (very small improvement)

### What Doesn't Work
- More tiles
- Different preprocessing
  - top 16 highest pixel deviation instead of darkness
  - stain normalization

### Final Solution
- Ensemble of ```swin_large_patch4_window12_384``` and ```coat_lite_medium```
- Some luck 🙏

### Some lessons learned from other competitions and applied here:
- Since public LB contains very few samples, we have to do a correct CV, which is what we could do and rely on.
- I tuned models not only to improve the average CV score but also to reduce the deviation of the 5-fold CV, so the model could perform more stable in an unseen test set.
- Implement the right loss and metric: I found a lot of public kernels using logloss as train loss and even evaluation directly instead of implementing competition metric and using it as a loss function for modeling. 
  - Logloss as evaluation behaves very differently from the competition metric.
  - Logloss as training loss cause worse competition metric in my validation.
- Try our best and hope for the best. 
  - We try our best to do CV correctly, to improve CV score while reducing uncertainty.
  - We hope for the best, since the dataset is not large enough to tell the difference between the last few digits, there will always be shakeup as expected. Keep an optimistic mind and move forward (Less painful for a bad shakeup)


My best CV with the highest mean and lowest deviations also achieve the best private scores.
[final selection and best private lb]

Submission Kernel:
https://www.kaggle.com/code/khyeh0719/mayo-submission
