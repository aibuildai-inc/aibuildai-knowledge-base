# 5th place solution - my part

Competition: hpa-single-cell-image-classification
Rank: #5
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238487

First of all, I would like to thank my fantastic teammates @its7171 and @tivfrvqhs5 . We realized 5 days before the deadline that we have to recalculate everything, and we managed to do so, selecting our submissions 1 hour before the deadline. The fact that it worked is a miracle.

Big congrats to @bestfitting who once again has shown his greatness, to the surprise of nobody :) We were suspecting you were #1 already for a long time, even when we were higher on public LB.

Congrats to all other teams - it was great to compete with you. 

Finally, I would like to thank the hosts for creating such an interesting problem for us to tackle.

I will share key parts of my solution, which brought the largest score boost. Other components of my models are fairly standard:

**1. Model on cell-level and progressive pseudo-labeling** 

I started with models trained on a whole image level, then I moved to models trained on a single-cell level. When assigning labels to single-cell images, I used the following approach:

```
threshold_std_above_mean = 0.5
threshold_pred = 0.9

for i in range(num_classes):
    cell_level_df[f'cell_label_class{i}'] = ((cell_level_df[f'gt_class_{i}'] == 1) 
                           & ( (cell_level_df[f'img_pred_rank_{i}'] == 1) 
                                   | (cell_level_df[f'std_from_mean_{i}'] > threshold_std_above_mean)
                                   | (cell_level_df[f'pred_class_{i}'] > threshold_pred)  ) ).astype(int)
```

The logic behind the above formula is the following:
I set the label for a single-cell image to 1 for a given class only if:
- The whole image has label 1 for this class
- This particular cell has the highest prediction for this class among all cells in the image, or is above 0.9 or is 0.5 standard deviations higher than the mean prediction for this class on this image

Those parameters were tuned using feedback from LB. I did 3 iterations -> models -> preds -> labels. This was the single biggest source of boost for my models. 

**2. Filtering our cells detected by segmentation model, but invisible to humans** 

When the blue channel is very weak, sometimes the official segmentation model provided by the hosts detects a cell, even when it is nearly invisible to the human eye, and could surely be removed by manual labelers. This is a simple condition I used, which brought like 0.04 improvement on the LB ( I assume blue is the 2nd channel):

```
cell_img[2,:,:][cell_img[2,:,:]>5] < 25 
```
Such cells were removed from the predictions

**3. Manual review of mitotic spindle**

I could not resist :) I spent a couple of evenings manually reviewing all images with mitotic spindle (label for class_11 == 1) and some high-predictions for class 0. This improved score of the mitotic spindle from 0.024 to 0.032. I can release this dataset if anyone is interested.
