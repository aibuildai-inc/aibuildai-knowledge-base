# Gold medal - 12 place solution

Competition: bengaliai-cv19
Rank: #12
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/135998

First I want to thank the organizers for an awesome competition. It was really challenging
My team got a good result, we have our first gold medal and most important than that, there were a lot of useful things to learn

And a quick overview of our solution

We used 3 different types of architectures  

**First Architecture Details:**
- 5 folds created with MultilabelStratifiedKFold  (mixed up by mean for final prediction)
- Image size: 128x128
- Augmentation: Cutmix (0.4) / Mixup (0.4), choosing by random one of them
- Label Smoothing
- Adam (lr=0.001, ReduceLROnPlateau(factor=0.8, patience=5))
- Trained for 100 epochs
- One head with 186 results than split the results into 3 softmax





**Second Architecture Details:**
- 5 folds created with MultilabelStratifiedKFold  (mixed up by mean for final prediction)
- Image size: original image
- Augmentation: Cutmix (choose random from 0.3, 0.4 ,0.5, 0.6, 0.7) + Cutout(max_holes=6, max_height=12, max_width=12, p=0.5 )
- Label Smoothing
- Adam (lr=0.001, ReduceLROnPlateau(factor=0.8, patience=5))
- Trained for 150 epochs


**Third Architecture Details**
- 2 folds created with MultilabelStratifiedKFold  (mixed up by mean for final prediction)
- Image size: 300x300
- Augmentation: Cutmix (choose random from 0.3, 0.4 ,0.5, 0.6, 0.7) + Cutout(max_holes=6, max_height=12, max_width=12, p=0.5 )
- Label Smoothing
- Adam (lr=0.001, ReduceLROnPlateau(factor=0.8, patience=5))
- Trained for 150 epochs


The final submission was the mean of all the folds from the 3 architectures  (12 folds) each with the same weight
The difference from public to private was 0.9784 -&gt; 0.9507

**Conclusions**
In the end, I can say that our solution did not use any unusual ace in the sleeve, not any unusual trick, just respecting the good practices in computer vision.
