# 10th place Code+Datasets (LB: 0.591) Mask R-CNN single model

Competition: data-science-bowl-2018
Rank: #7
Source: https://www.kaggle.com/c/data-science-bowl-2018/discussion/56238

Note: (currently 7th place while leaderboard is being updated) 

## TLDR:  Resources related to this project
* [Data used to build the models](https://www.kaggle.com/gangadhar/nuclei-segmentation-in-microscope-cell-images)
* [Code](https://github.com/gangadhar-p/NucleiDetectron)
* [Detailed Writeup](https://github.com/gangadhar-p/NucleiDetectron)
* [Pretrained model, predictions and visualization of submission](https://www.kaggle.com/gangadhar/nuclei-detectron-models-for-2018-data-science-bowl)

## Preview of training data
<div>
  
  <p></p>
</div>


## Dataset preparation
* There were several nuclei datasets with outlines as annotations.
   * Applied classical computer vision techniques to convert ground truth from outlines to masks.
   * This involved adding boundary pixels to the image so all contours are closed.
   * Given outlines of cells with overlaps/touching or at border,
      * Mark an outer contour to encompass contours that are at image edges.
      * then do cv2.findContours to get the polygons of mask.
      * Ref [parse_segments_from_outlines](https://github.com/gangadhar-p/NucleiDetectron/blob/master/lib/datasets/nuclei/mask_encoding.py#L184)
* Standardized all datasets into COCO mask RLE JSON file format.
   * You can use [cocoapi](https://github.com/cocodataset/cocoapi) to load the annotations.
* Cut image into tiles when images are bigger than 1000 pixels
   * This was necessary since large image features did not fit in GPU memory.

## Preprocessing
* Cluster images into classes based on the color statistics.
* Normalize classes size
   * Oversample/undersample images from clusters to a constant number of images per class in each epoch.
* Fill holes in masks
* Split nuclei masks that are fused
   - Applied morphological Erosion and Dilation to seperate fused cells
   - Use statistics of nuclie sizes in an image to find outliers
*  [ZCA whitening of images](http://ufldl.stanford.edu/wiki/index.php/Whitening)
*  Zero mean unit variance normalization
*  Grey scale: [Color-to-Grayscale: Does the Method Matter in Image Recognition](http://tdlc.ucsd.edu/SV2013/Kanan_Cottrell_PLOS_Color_2012.pdf).
   - Very important how you convert to grey scale. Many algorithms for the conversion, loss of potential data.
   - Luminous
   - Intensity
   - Value: This is the method I used.
*  [Contrast Limited Adaptive Histogram Equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)

## Augmentation

Data augmentation is one of the key to achieve good generalization in this challenge.

### Training time augmentation

* Invert
  * This augmentation helped in reducing generalization error significantly
  * Randomly choosing to invert caused the models to generalize across all kids of backgrounds in the local validation set.
* Geometric
  * PerspectiveTransform
    * This is very useful to make the circular looking cells to look stretched
  * PiecewiseAffine
  * Flip
  * Rotate (0, 90, 180, 270)
  * Crop
* Alpha blending
  * Create geometrical blur by affine operation
  * Shear, rotate, translate, scale
* Pixel
  * AddToHueAndSaturation
  * Multiply
  * Dropout, CoarseDropout
  * ContrastNormalization
* Noise
  * AdditiveGaussianNoise
  * SimplexNoiseAlpha
  * FrequencyNoiseAlpha
* Blur
  * GaussianBlur
  * AverageBlur
  * MedianBlur
  * BilateralBlur
* Texture
  * Superpixels
  * Sharpen
  * Emboss
  * EdgeDetect
  * DirectedEdgeDetect
  * ElasticTransformation


### Test time augmentation
1. Invert: Have improved the performance a lot
2. Multiple Scales 900, 1000, 1100
3. Flip left right


## Architecture changes to baseline Detectron

Detectron network configuration changes from the baseline e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml are:

1. Create small anchor sizes for small nuclei. RPN_ANCHOR_START_SIZE: 8 # default 32
2. Add more aspect rations for nuclei that are close but in cylindrical structure. RPN_ASPECT_RATIOS: (0.2, 0.5, 1, 2, 5)
3. Increase the ROI resolution. ROI_XFORM_RESOLUTION: 14
4. Increase the number of detections per image from default 100. DETECTIONS_PER_IM: 500

## Training
1. Decreased warmup fraction to 0.01
2. Increased warmup iterations to 10,000
3. Gave mask loss more weight WEIGHT_LOSS_MASK: 1.2

## Segmentation Post processing
  * Threshold on area to remove masks below area of 15 pixels
  * Threshold on BBox confidence of 0.9
  * Mask NMS
    * On decreasing order of confidence, simple union-mask strategy to remove overlapping segments or cut segments at overlaps if overlap is below 30% of the mask.

## What worked most
1. Inversion in augmentation
2. Blurring and frequency noise
3. Additional datasets, even though they caused a drop on the public leaderboard, I noticed no drop in local validation set.

## What did not work
1. Mask dilations and erosions
   * This did not have any improvement in the segmentation in my experiments
2. Use contour approximations in place of original masks
   * This did not have any improvement either. Maybe this could add a boost if using light augmentations.
3. Randomly apply structuring like open-close
4. Soft NMS thresh
   * Did not improve accuracy
5. Color images
   * Did not perform as well as grey images after augmentations
6. Color style transfer. Take a source image and apply the color style to target image.
7. Style transfer: Was losing a lot of details on some nuclei but looked good on very few images.
8. Dilation of masks in post processing, this drastically increased error because the model masks are already good.
9. Distance transform and split masks during training.

## Things I didn't have time to try
1. Ensemble multiple Mask R-CNN's
2. Two stage predictions with U-Net after box proposals.
3. Augmentation smoothing during training
   * Increase the noise and augmentation slowly during the training phase, like from 10% to 50%
   * Reduce the augmentation from 90% to 20% during training, for generalization and fitting.
4. Experiment with different levels of augmentation individually across, noise, blur, texture, alpha blending.
5. Different layer normalization techniques, with batch size more than one image at a time. Need bigger GPU.
6. Little bit of hyperparameter search on thresholds and network architecture.

## Things I did not think of
U-Net with watershed, did not think this approach would outperform Mask R-CNN

## Acknowledgements:
Kaggle community was a great source of inspiration and the discussions are very useful. Special thanks to Discussion Gradmaster [hengck23](https://www.kaggle.com/hengck23).

## Code References
- [Detectron](https://github.com/facebookresearch/detectron).
  Ross Girshick and Ilija Radosavovic. Georgia Gkioxari. Piotr Doll\'{a}r. Kaiming He.
  Github, Jan. 2018.

- [Image augmentation for machine learning experiments](https://github.com/aleju/imgaug).
  Alexander Jung.
  Github, Jan. 2015.

- [Normalizing brightfield, stained and fluorescence](https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence).
  Kevin Mader.
  Kaggle Notebook, Apr. 2018.

- [Fast, tested RLE and input routines](https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines).
  Sam Stainsby.
  Kaggle Notebook, Apr. 2018.

- [Example Metric Implementation](https://www.kaggle.com/wcukierski/example-metric-implementation).
  William Cukierski.
  Kaggle Notebook, Apr. 2018.
