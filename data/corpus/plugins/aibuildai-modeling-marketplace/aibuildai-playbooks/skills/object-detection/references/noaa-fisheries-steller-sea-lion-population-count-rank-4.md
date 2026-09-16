# Solution in a nutshell. 2th Public LB / 4th Private LB

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #4
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35442

Here I will give a short description of the approach of our team (me and [DmitryKotovenko][1]).  
The source code is available on GitHub https://github.com/asanakoy/kaggle_sea_lions_counting.


Preprocessing
--------------

Thanks to @Radu Stoicescu for his blob detection to get corrected counts from dotted images.

GT count for each tile was generated as a sum over heatmap (to overcome cases with lions on the border of the tile).   On top of each lion we put a Gaussian with a standard deviation heuristically estimated by calculating the smallest distance between lion on the image.   
We set the standard deviation to be 50 at least for each Gaussian and adjust it according to size of the animals from different classes (
multiplied by 2 for adult males and by 0.5 for pups).

<a href="https://ibb.co/hhAda5">GT Gaussians for tile</a>
![GT Gaussians for tile][2]


Model and Training
------
Our model incarnates **regression for 5 classes on tiles of the images**. (In similar spirit as the approach of @outrunner)

**Inception Resnet v2** pretrainedon Imagenet.   
We substituted the last layer with 256-way FC layer + dropout + 5-way FC layer on top. + RMSE loss.
Then we fine-tuned the model on 299x299 image tiles with Adam optimizer. 

**Augmentations:**  random rotation on 90/180/270 grads, random flip left-right, bottom-up.

**Scale augmentations:** one model without them, one model with 0.83-1.25 random scaling, one model with 0.66 - 1.5 random scaling.

<a href="https://ibb.co/kVhwTQ">RMSE on val for 3 best models</a>
![RMSE on val for 3 best models][3]
    

Testing
-------

**During test** we made predictions up to 5 times for each model using different shifts of the tiles in the image. 
Test images were downscaled in 0.4-0.5 times.

The final ensemble was made by averaging all the predictions.   
**Private LB RMSE:** *13.18968*  
**Public LB RMSE:** *13.29065*

Applying further postprocessing as suggested by @outrunner, could improve results.  
Just increasing the number of pups by 20% gives a huge improvement:  
**Private LB RMSE:**  *12.58131*  
**Public LB RMSE:** *12.75510*


Some negative experiments
-------

We labeled some images from train set according to scale and trained a CNN to regress a scale of the image.   This could unify all the images to have the same approximate size of the lions of corresponding classes and simplify the CNN training to count animals.

But it didn't work out.   I reckon, the reason is the high variation in terrain and inability to estimate scale of objects if you look at them within a small spatial context (even with my own eyes).

  [1]: https://www.kaggle.com/chelovekparohod
  [2]: https://preview.ibb.co/bJ5Bv5/Screenshot_from_2017_06_28_16_28_57.png
  [3]: https://preview.ibb.co/jx6O8Q/Screenshot_from_2017_06_28_17_02_17.png
