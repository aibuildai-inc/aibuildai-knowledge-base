# Our solution, 4th place on the private LB

Competition: data-science-bowl-2018
Rank: #4
Source: https://www.kaggle.com/c/data-science-bowl-2018/discussion/55118

In our team we have evaluated both UNet and Mask-RCNN based solution, but for us Unet worked significantly better so we used Unet based model for submission.

The UNet based solution is inspired by the Deep Watershed Transform paper: https://arxiv.org/pdf/1611.08303.pdf


For each pixels, we predicted the x,y components of vector pointing from the instance border like described in the DWT paper and predicted the mask, watershed levels and nuclei centers using the second connected UNet. 

Attached the diagram with our model

![Model description][1]

Overall Unet predict the mask pretty well but it was necessary to find a way to reliably segment nuclei.

Approaches we tried
================

The first approach was to predict using the single UNet model:

 * Mask, BCE + DICE loss
 * Nuclei centers with 3x3 patches around the center of mass as a training label, BCE loss
 * Area of nuclei used to normalize loss from vectors for large and small nuclei
 * X,Y of vector to the center of nuclei, MSE loss normalized by nuclei area


For touching nuclei the vector value to the centers changes sign, so it changes sharply and the loss is the biggest on the nuclei border which forces model to learn to separate instances. The postprocessing was quite straightforward:

 1.  Find the centers of nuclei using predicted centers output, expecting the area of each prediction to be approx 9.0 (matring area of 3x3 training patch)
 2.  For each pixel in predicted masc, assign it to the cluster nearest to position predicted vector to the center points to.

This approach worked and scored over 0.5 on the public leaderboard.


Improvement:
-----------------

The second approach was to try the deep watershed transform idea to predict watershed energy levels (mask eroded by different offsets) by predicting intermediate unit vector fields pointing from the nearest border pixel. Instead of training 3 independent models for segmentation, vector field and watershed energy predictions I used the single UNet to predict everything. With the seeds we used the continuous areas after applying the threshold over the sum of energy levels, in the similar way to the DWT paper. The result was slightly better.

Improvement:
-----------------

Use the predicted centers as seeds for watershed transform instead of the energy level with threshold. The score improvement was more significant comparing to switching to DWT.

Improvement:
-----------------

Predict only vector fields with the first UNet, concatenated predicted fields with the last layer used to predictions of all the other fields using another UNet. This has little to no impact on predicted masks and the energy levels but helped to significantly improve the quality of predicted nuclei centers. 

Improvement:
-----------------
Simple mean ensembling of 8 TTA flips/rotations had very small but consistent improvement.

Improvement:
-----------------

Better postprocessing, the idea - since for prediction the center values we are using the 3x3 path of area 9, we can expect the total area of predicted center to be close to 9 as well, even for cases of complex connected nuclei hard to predict. 
This allowed to do following post processing improvements:

 * After watershed transform from detected centers, check for missing large masks with the total center prediction &gt; ~5.0, and add them.
 * Instances with the total center value integral &gt; 9.0 * 1.5 are most likely have two nuclei connected but with very hard to predict centers. Split centers to two clusters using KNN and re-run watershed.
 * We tried the similar approach to topcoder team of estimating IOU using information about the shape and other predicted instance properties and run optimiser for the score to decide if particular instance should be included to submission or not, but the quality of IOU prediction was not sufficient for this to work reliably. Instead we calculated IOU between masks of the ensemble and individual TTA variants. If the median IOU is low, it means different models predicted different shapes and it’s better to exclude such mask from submission to avoid penalty at high IOU thresholds.


Extra data used
-------------------

We added extra annotated datasets listed in the forum thread and annotated some images, mostly color histology images. We also used synthetic data to generate more cases of touching and overlapped nuclei.

Final model details
==============

Attached an example of the early model predictions, including predicted vectors, centers and energy levels.

![Example prediction][2]

We tried to use the imagenet pretrained models as UNet encoders but the result was the same or worse comparing to UNet trained from scratch.

For unet encoders we used 
Conv2d - BN - Relu - Conv2d - Relu
Decoders:
Upsample/concatenate - Conv2d - Relu

We used quite a large number of filters (64/128) even on the high resolution levels as model had to predict many outputs but increased number of filters less than twice as we did not have as much complex semantic informations as with models trained on imagenet.

Unet1 was 6 levels deep (decreased the resolution 2x 6 times) while Unet2 was 4 levels deep. 

As input we used B/W image or H channel of HED stain decomposition. H worked slightly better for histology images but slightly worse for some other stains. For the final submission we ensembled 4 models trained on B/W images and 2 on H channel.

We trained on 256x256 patches with significant level of augmentations and predicted on 1024 pixels tiles with 128 pix overlap on each side. We used the “SAME” padding and relied on the large tiles padding and overlap to avoid corner effects.

At the end of competition we tried to train a model predicting only data used for post processing without vector fields and the result was significantly worse. Significantly reducing the loss of vectors to the center of nuclei had a little impact to the results, so most likely vectors from corners are more important. Even while not directly used, it helped to better predict centers and watershed energy levels.

[Solution source code][3]


  [1]: https://kaggle2.blob.core.windows.net/forum-message-attachments/317711/9258/Vector%20unet.png
  [2]: https://kaggle2.blob.core.windows.net/forum-message-attachments/317711/9259/nuclei_descr.png
  [3]: https://github.com/pdima/kaggle_2018_data_science_bowl_solution
