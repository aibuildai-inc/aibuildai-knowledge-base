# Post Competition Architecture Discussion

Competition: passenger-screening-algorithm-challenge
Rank: #1
Source: https://www.kaggle.com/c/passenger-screening-algorithm-challenge/discussion/45805#261052

My final submission was a little complicated, but basically it boils down to 5 variations on the strategy outlined below.  The differences were mainly in which pre-trained ImageNet model was used, what amount of augmentation was applied, size of the threat zones, and what ended up in the 3rd color channel.

## Model Input
Of the various formats, only the APS was used.  Neither of the A3D formats seemed to add anything and generating an image from the AHI is still a mystery to me.

With only ~1,200 scans and 23ish subjects, augmentation seemed critical for a model to generalize to new subjects. For me this involved rotations, translations, contrast and brightness adjustment, horizontal reflection, and gaussian blur.  In addition, ~800 more scans were generated from the training scans by using GIMP.  Mostly this involved transforming and transplanting a threat object from one location to another or from one scan to another.

The scans are all monochrome whereas the pre-trained ImageNet models accept three channel color input.  Triplicating the the monochrome scan for each channel was one option, but it seemed like a waste.  To make use of the multiple scans per subject, I took the average and standard deviation per pixel of ten similar scans of the subject and placed them in the other two color channels.  This had the aesthetically pleasing effect of making the threats stand out in a nice red color.
![model_input][1]
Of course the solution needed to be fully automated, so selecting similar scans needed to automated as well.  The easy way is to take the average absolute difference per pixel between each of the ~1,200 scans.  Whichever ten scans have the smallest distance could be used.  I found that a VGG-style deep autoencoder worked a little better, but the idea is essentially the same. 

## Model Design
Initially I had started with a 17-target MVCNN similar to what Moejoe (Shayan) described in this thread, but eventually I found that splitting the image up into four overlapping zones worked a little better.  Potentially the improvement was due to using a higher resolution on each of the smaller threat zones or possibly due to eliminating irrelevant information from other parts of the scan.  In any case this split the model into four components roughly aimed at: Arms (1-4), Chest (5-7,17), Waist (8-12), and Feet (13-16).
![model_zones][2]
Each of the four component models had a similar design.  They took 8 of 16 equally spaced images from the whole APS scan and fed them into a pre-trained ImageNet model, resnet-50 for instance, whose weights were shared over the different inputs.  Given that we are asking the model to not just predict the threat's existence but also the location, we want to preserve spatial information so in contrast to MVCNN, which takes the maximum of all the outputs, here we just concatenate the outputs together.  This leads to a very large number of channels which need to be reduced using a 1x1 convolution.  After that we have a very standard top design with a few dense layers.  Some small amount of dropout was needed in the last few layer to prevent the model from overfitting.
![model_diagram2][3]
## Prediction/Calibration
Test time augmentation was helpful here.  All the augmentation applied to the training images was applied here at three checkpoints for each of the models.  It was probably overkill, but predictions were made for each zone model on each scan around 100 times.  All the various predictions were just averaged.

Calibrating predictions to generalize well to new subjects was also helpful.  For this a 5-fold cross-validation was performed, with scans from any particular subject all in the same fold.  Using the out-of-sample scores from cross-validation and the original labels before any corrections, a LightGBM model was built.  This LightGBM model was then applied to the stage-2 predictions.


  [1]: https://kaggle2.blob.core.windows.net/forum-message-attachments/261052/8101/model_input.png
  [2]: https://kaggle2.blob.core.windows.net/forum-message-attachments/261052/8102/model_zones.png
  [3]: https://kaggle2.blob.core.windows.net/forum-message-attachments/261052/8103/model_diagram2.jpg
