# 4th place solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #4
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539443

Congrats to all prize and medal winners! This year's RSNA competition required us to carefully handle data and build a pipeline, which was a lot of fun. We share our solution.

# Summary
Our solution detects the keypoint, which is the region of interest in the symptom, and builds a classification model using the surrounding crops as input.
The results of each model are refined by the stacking model and submitted as the final result.


# Keypoint detection Model
## Disk Level detection model and Keypoint detection for Axial, Sagittal T1/T2 (@yu4u)

Resize each axial slice to 128×128 and use a 2.5DCNN + LSTM model to estimate which level (L1, L2, …, S1) each slice belongs to. Subsequently, detect the boundary slices between each level. From these slices (up to five), use a UNet model to detect the left and right keypoints.
Similarly, resize the sagittal T1 slices to 128×128 and use a 2.5D CNN model to identify the left and right slices belonging to the foraminal zone where keypoints should be detected. Then, individually detect keypoints for five levels from these left and right slices.
For sagittal T2/STIR, simply extract the middle slice of the series and detect keypoints for the five levels.


## Keypoint detection model for Sagittal (@tattaka)

We resized each of the Sagittal T1, T2/STIR images to 20x256x256 and predicted the xy coordinates of the keypoints.
The xy coordinates of the keypoints were taken from the shared [Lumbar Coordinate Dataset](https://www.kaggle.com/datasets/brendanartley/lumbar-coordinate-pretraining-dataset).
For the backbone, we used caformer_s18, convnext_tiny, resnetrs50, and swinv2_tiny, and applied SCSE attention to the UNet Decoder.
As for the loss function, we used BCELoss * 0.2 + DICELoss * 0.8.

# Classification model

## Multi-view input, multi-condition output model (@tattaka)


We crop the areas around the keypoints inferred from the volumes of Sagittal T1, Sagittal T2/STIR, and Axial T2, and use them as inputs to classify the conditions at each level.
Each image is cropped to a size that is twice the distance between the neighboring keypoints. For sagittal images, padding is applied if all slices are fewer than 30, and linear interpolation is used if there are more. (There is also a model variation that simply uses linear interpolation to resize to 20.)
For axial images, slices are taken from the range of ±2 around the predicted gaps between the discs.
After the images are input into a 2D model backbone, features are extracted from each slice, and the final output is obtained using a transformer encoder and attention pooling on the extracted features. 

The model used for the final submission includes variations such as:
* A model that processes cropped slices with one or two backbones,
* Multiple augmentation patterns,
* Various preprocessing patterns for multiple slices.

The backbones used were caformer_s18, resnetrs50, [rdnet_tiny](https://github.com/naver-ai/rdnet), and maxxvitv2_nano.
A key technique to successfully train this model is to apply attention pooling to the features before inputting them into the transformer and calculate the auxiliary loss.

## Single-view input, single-condition output model (@yu4u)



In this part, severity score is estimated via cropped images.

For Sagittal T1 and Sagittal T2/STIR images, the cropping scale is determined based on the average distance between the keypoints of the five levels, and patches are cropped centered on the keypoints. For Axial T2 images, an affine transformation is applied to position the left and right keypoints at specific locations within the patch before cropping. Subsequently, the cropped images are input into a 2.5D CNN model to calculate the severity score. For Axial T2 images, the model is used not only to predict subarticular stenosis but also spinal canal stenosis. Other combinations did not yield significant results.


# Ensemble and stacking

## Nelder-Mead guided stacking MLP
We constructed a stacking model using an MLP.
The key feature of this model is that, in addition to the standard skip connection, the output optimized by the Nelder-Mead method is added to the model output. (In other words, the model learns the difference between the ground truth and the Nelder-Mead results.)
The inputs for ss, scs, and any consist only of the results from their respective classification models, while nfn is fed the concatenated outputs of scs, ss, and nfn.

## Stacking LightGBM and XGBoost
The outputs of the individual models are stacked using LightGBM and XGBoost. In this stacking approach, the same model is used separately for each level, and only inputs of the same type as the output target are utilized. As a result, the input dimensions are equal to the number of models multiplied by three. Using predictions for different targets was not effective.

# Source code and notebooks
## source code
* tattaka's part: https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public 
* yu4u's part: https://github.com/yu4u/kaggle-rsna2024-4th

## notebook
* best submission notebook: https://www.kaggle.com/code/ren4yu/rsna2024-ensemble-submission-stacking-tattakav2/notebook?scriptVersionId=200046802
* MLP stacking for scs: https://www.kaggle.com/code/tattaka/rsna2024-stacking
* MLP stacking for nfn: https://www.kaggle.com/code/tattaka/rsna2024-stacking-nfn
* MLP stacking for ss: https://www.kaggle.com/code/tattaka/rsna2024-stacking-nfn
