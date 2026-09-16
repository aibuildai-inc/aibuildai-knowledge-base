# LB0.9972 - 5th Place Overview

Competition: carvana-image-masking-challenge
Rank: #5
Source: https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40144

Overview
--------
My overall pipeline looks something like this:

![Carvana Pipeline][1]

As per above I only used a pretrained Resnet-50 based FCN for this competition (see "Future Improvements" for why), fine tuning all layers (Adam, learning rates from 1e-4 -&gt; 1e-6).  Scales of 1280x1280, 1600x1280, 1918x1280, and 2010x1340 were used to generate different 7 different weighted models for ensemble.  In summary:

 1. 1918x1280(HQ, TTA6x) [weight=0.2,public LB=0.9970+, private=0.9969]
 2. 1918x1280(TTA6x) [weight=0.3,public LB=0.9970+,private=0.9969]
 3. 1918x1280(HQ, TTA2x), Fold 2 [weight=0.2,public LB=0.9970+,private=0.9969]
 4. 1600x1280(TTA4x) [weight=0.1,LB=0.9969]
 5. 2010x1340(TTA2x) [weight=0.1,LB=0.9969]
 6. 1280x1280(TTA6x) [weight=0.05,LB=0.9968]
 7. 1280x1280(TTA6x, using only dice loss) [weight=0.05,LB=0.9968]

(1918x1280 and 2010x1340 scales need &gt;8GB cards)

**Augmentations** Horizontal flips, rotations (up to 10 degrees), height/width translation of about 5%, and zooms of about +/-10%.  There was a strong penalty on using strong augmentations - I turned down from an initial rotation of 30 degrees and I remember local and LB improving by a factor of 0.0001.  Did not sweep further or try other augmentations (contrast/brightness, etc.)  TTA used both horizontal flips and slight vertical translations (up to 6x permutations).  

**CV** CV=0.9970 for 1918x1280, 0.9969 for 1600x1280/2010x1340, and 0.9968 for 1280x1280, and their CVs were close to LB.  I had some earlier runs with random splits between (not car split), and these seemed to give a slightly higher CV than LB (e.g. CV of 0.9971+ would yield a LB of 0.9970), but kept them anyway since I didn't want to retrain everything again.

**Loss** binary cross entropy - log(dice).  This gave slightly better results in local score than dice only.  Batch size of 1 was used.  Did not try weighted losses.

Post-processing
---------------

I spent the bulk of my time (while waiting for models to complete training) to implement post-processing.  Like team "David" I saw a lot of small artifacts representing the antenna post-prediction, and found a way to merge (first check if the artifacts represent a straight line [check their R^2], then train a linear regressor and add a point inside the main hull, then create a convexHull around these artifacts + thresholding to improve the shape).  Although the process was complex, the score impact was minor but it helped.  

I also had another step to close all holes inside the main contour (with certain distance from the edge, to avoid closing sidebars or wheels), and also close deep U-shaped concave contours (U-shaped only, to avoid closing say the front/back of a pickup).  This helped significantly in some cases especially on the van, and I believe that this helped a lot for private LB.

Finally, one more post-processing step was added to remove artifacts some distance away from the main contour.  I had two thresholds - one to determine a far away but small artifact/contour (to avoid purging unmerged antennas), and a close threshold to determine large objects close by, and purge them.

What was the impact of these fixes? Looking the private LB showed an improvement from 0.997010 (before post-processing) to 0.997209 (after post-processing - a big jump!

In addition, I ran a visual check on the test predictions using the following criteria:

 1. Stability of ensembles (i.e. for each model, the pixel sum variance)
 2. Number of contours (i.e. if too many contours, prediction is dirty)
 3. Area of the second largest contour (i.e. same idea as #2, if there is a second contour that is large, something is off)

This helped to catch some issues like the van, the single blue car (which was noisy for any architecture/scale including UNET), etc.

Other Notes
-----------

Earlier on I stopped all progress once I found out the LB was broken and resumed only once it was fixed :) (so a few weeks lost there).  Also, I only purchased a GTX1080Ti about 3 weeks before the end of the competition - this was required to train scales &gt; 1600x1280 for the pretrained model.

Hardware Used
-------------------

 1. GTX1080Ti - purchased 3 weeks prior to end of competition, so not much mileage.  Needed for 1918x1280 and 2010x1340.
 2. GTX1080 
 3. GTX1070 - inferencing / TTA only

Future Improvements
-------------------

I only used a Resnet-50 FCN here (rather boring, yes), since from a quick experiment I saw comparatively better performance for the pretrained Resnet vs UNET at similar scales (but without trying to crop or layer tune further on the UNET).  

In retrospect - that was not a good decision since my ensembles turned out to be so-so (without good diversity), but I ran out of time with my limited hardware to attempt other architectures.

Also, I tried to use Resnet-101/152 as well but these ran out of memory beyond 1024x1024, and fine tuning only certain layers (e.g. deep layers) did not give improvements over all layers in Resnet-50.


  [1]: https://kaggle2.blob.core.windows.net/forum-message-attachments/225124/7421/CarvanaPipeline.jpg
