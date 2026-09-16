# 18 place write up  (short meditation on results)

Competition: imaterialist-fashion-2019-FGVC6
Rank: #18
Source: https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/discussion/95238#latest-566318

This is our short meditation on our results at strategy:

1.  r101 mask rcnn 1024x1024 (1 fold, it took 5 days  on our machine)

2. ansemble predictions from multiple snapshots of r101 mask rcnn (select prediction with highest confidence, when multiple predictions overlap)

3. Train several multiclass classifiers (xception,resnext,densenet) two detect if particular object classes present on image. Fins optimal treshold for discarding predictions from mask rcnn. [Our Classification Pipeline](https://github.com/musket-ml/classification_training_pipeline)

4. Train a lot of segmentation networks at least one per class to refine masks from mask rcnn ([Segmentation Pipeline](https://github.com/musket-ml/segmentation_training_pipeline)) -&gt; This was our main source of improvements.

At the same time we were desperately trying to find solution for attributes. Initially nothing worked. But later we have used following approach:

Take a mask crop and a full image, and feed them in two independent inputs of image classifier, with multiple outputs per independent attribute groups (we have found them by analizing attribute co occurances, and assuming that sometimes coocurences labeling errors).

Train a lot of such classifiers, then take only those objects for which 75% of classifiers agree with the result of blend and also when most of them are doing confident predictions.

This gave us very slight improvement of our score (~0.00500 in total) 

**Errors:**

- Spending our pretty limited resources on attributes was a big error.
- We did not actually realised and used full potential of mask rcnn :-(. Our initially high place on leaderboard, gave us false feeling that we used most of its potential. 

**Resources**

We have used: 2x1080Ti on my machine, 2x1080Ti on Denis machine  and 1080 on Konst machine. 


Regards,
Pavel
