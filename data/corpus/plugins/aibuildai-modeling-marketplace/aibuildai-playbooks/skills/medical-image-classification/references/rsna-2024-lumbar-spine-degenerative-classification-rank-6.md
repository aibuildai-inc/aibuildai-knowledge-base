# 6th Place Solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #6
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813

Thanks to kaggle and everyone involved for hosting such an interesting competition. We learned a lot about MRI data and how to create strong models for it. 
This competition was very challenging (and time-consuming !) because it had 3 underlying modalities (SCS, NFN, SS) and 2 tasks (disk localization, severity classification).

## TLDR
Our solution is an ensemble of multiple models which train on study or series level for each individual intervertebral disc level. Crops of the original MRI are isolated in a first stage and the severity of the condition is predicted using the sequence of crops. Final results are aggregated using an MLP model that directly optimizes the competition metric.

## Cross validation 

A common cross validation approach was used within the team with four folds. Below is the tracking of CV against leaderboard scores. Our experience from previous RSNA competitions led us to expect a small yet reasonable shake-up, and we trusted our CV more than public LB.

<a href="https://ibb.co/LzxRtby">[cvlb]</a>

## Data & Augmentations
All data was sourced from competition data, except for the spinenet model weights. Some of our models used @brendanartley publicly shared coordinates (thanks a lot!). 

For preprocessing we mostly used min-max normalisation, and sometimes added windowing based on dicom window width and window length. We relied on albumentations for augmentations, and used a variety of approaches for the different pipelines:

- **Model 1 :**
  - ShiftScaleRotate, ElasticTransform
  - RandomGamma, RandomBrightnessContrast
  - MotionBlur, GaussianBlur
  - MixUp for the Sagittal only models.
  - Flipping left / right and the associated targets for some models.
- **Model 2 :**
  - ShiftScaleRotate
  - RandomBrightnessContrast
  - CoarseDropout
  - RandomCrop
  - For localisation of crops, augmenting the target coord position was very effective
  - Random shift start and end instance number of crop 
- **Model 3 :**
  - ShiftScaleRotate
  - RandomBrightnessContrast
  - CoarseDropout
  - RandomCrop
  - ChannelDropout

## Models

### Model 1

The code to this part of the pipeline is available on github
> https://github.com/TheoViel/kaggle_rsna_lumbar_spine

It achieves ~0.42 private LB, using only the sagittal data.

<a href="https://ibb.co/kcm5KVW">[RSNA-spine-theo-drawio]</a>


**Coordinates model :** A simple `coatnet_rmlp_2_rw_384` is trained on all the center frames of the sagittal images to predict the 5 (x, y) coordinates associated with the disk injuries. It is trained with the MSE on 10 classes, and used to generate crops. Crop generation is done by selecting the square of 20% of the image size around the predicted ROI center.

**Classification models :** On the crop above, we once again trained CoAtNets, this time `coatnet_1_rw_224` and `coatnet_2_rw_224` and with a RNN layer to incorporate the 3D information. 
We initially wanted to train separate models for each injury since each image modality had its target. For instance the SCS only models are trained to predict the 3 severity classes on the Sagittal T2 images, by sampling the 5 (or 3) frames at the center of the stack. But handling the two sagittal modalities together worked better, and even showed decent performance on the SS task. Further improvements came from adding more frames (5 is not enough to capture the SCS and SS signal) and using 3 RNN heads that had access to different frames - for the left, right and center (scs) targets. We also added MixUp and trained models for 10 epochs (vs 5 for the SCS models) and with a higher learning rate (1e-3 vs 5e-4 for the SCS models). 

**MLP :** The classification models are trained with the CE, and tweaked to maximize the AUC. The MLP model is here to aggregate predictions, and account for the competition metric. Surprisingly, what worked best here is to consider each target independently, i.e. the scs_l1_l2 does not interact with the scs_l2_l3 features nor the nfn_left_l1_l2 features in the MLP. The logits layer weights for the different levels are shared though, i.e. the model consists of 3 MLP (one for SCS, one for SS and one for SCS).

### Model 2

The code to this part of the pipeline is available on github
> https://github.com/darraghdog/kaggle-rsna-2024-6th-place-solution-model2/

<a href="https://ibb.co/2hCR7GR">[RSNA-2024-RSNA-2024-Lumbar-Spine-Degenerative-Classification-pptx]</a>

For Sagittal and Axial images we train severity classification on series and individual intervertebral disc level. Therefore we need to isolate each individual disc within the dicom. 

**Axial xy-localisation :** We use the train coordinates file to learn the xy-location of the labelled point for each slice. A backbone of `efficientnetv2_rw_t` with a linear head and L1-loss is used with learning rate 1e-4 for 16 epochs with a batchsize of 16. The dicom images are resized to 384 and images are fed to the model individually. 

**Sagittal xy-localisation :** We use spinenet to predict the left top corner point of the intervertebral disc. Many of the spinenet points are shifted incorrectly by one disc. We use the train coordinates file to identify the shifted ones and shift them by one disc. We then retrain the localisation on the corrected spinenet predictions. We exclude from training series where the label is too far from the spinenet prediction. The same training model and procedure is used as axial xy-localisation, except we have an xy label for each vertebrae and we add a mask label to indicate slices with no spinenet xy prediction (in this case, xy-loss is masked).

**Sagittal z-localisation :** We use the train coordinates file to learn the z-location (the annotated instance number). For all series we predict the annotated instance of spinal_canal_stenosis as well as left and right neural_foraminal_narrowing instance number. For sagittal t2, we mask the foraminal loss and for sagittal t1 we loss the spinal canal loss. L1 loss is used where the target is the number of annotations on the instance, between 0 and 5. Same training procedure as above.

**Axial z-localisation :** We leverage sagittal point to axial level mapping from the repo M-Scan repo - code here - to map the xyz position of the sagittal t2 series to the z-position of the patient’s axial series. This identifies which instance number, or slice, contains which intervertebral disc. Inspiration from Ian Pan, Heng and others who shared this approach earlier on in the competition.

**Axial severity classification :** Crops for stage 2 are made by taking the distance of right to left annotation and extending outward either side by half this distance. On the z-position the sequence is cropped by finding the max distance from one level to the next and extending outward either side by this number of slices from the level’s center slice. Therefore we use a variable number of slices per series/level. A 2.5d model of `efficientnetv2_rw_t` with a bidirectional single layer RNN head (512 dim) and weighted CE loss is used. We use learning rate 4e-4 for 5 epochs with a batch size of 8 series/levels with all crops resized to 256 dim.
We use two separate axial models. One to predict the Axial t2 severity, and one to predict the Sagittal t2 severity. The model was not effective in predicting Sagittal t1 severity. 

**Sagittal severity classification :** Crops for stage 2 are made by taking the max distance of one vertebrae to the next and extending outward either direction for each level from the levels center xy position. On the z-position the sequence is cropped by excluding slices which were predicted to have no spinenet annotation (as seen above under Sagittal xy-localisation above). Sagittal t1 and t2 labels and series were trained together.  

### Model 3 

The code to this part of the pipeline is available on github
> https://github.com/darraghdog/kaggle-rsna-2024-6th-place-solution-model2/

<a href="https://ibb.co/WxgRc9H">[Screenshot-2024-10-19-at-09-45-26]</a>

**Axial xy-localisation :** We use the train coordinates file to learn the xy-location of the labelled points for each slice. A backbone of `tf_efficientnetv2_s` with a linear head and L1-loss. The dicom images are resized to 384 and images are fed to the model individually. 

**Sagittal xy-localisation :** We use the train coordinates file to learn the xy-location of the labelled point for each slice. A backbone of `tf_efficientnetv2_s` with a linear head and L1-loss is used. The dicom images are resized to 384 and images are fed to the model individually. It was beneficial to train a single model on a combined dataset of T1 and T2 sagittal images 

**Axial z-localisation :** Same approach as Model 2. We derived axial z localisation for each vertebrae from T1/ T2 Sagittal xy-localization from same study using 3d coordinates. 3d coordinates are derived separately for T1 and T2 and then averaged. 

**Axial Crops :** Multiple axial series for a study are combined by sorting by ImagePositionPatient. Then median of x1/y1 and x2/y2 predictions for all slices are used to create 112x224 sized bounding boxes for a study. For each slice of the axial series a intervertebral disk (IVD) is assigned using the closest derived z coordinates. A maximum distance of 20 was used. So for each IVD a 3d bounding box with height x width of 112x224 and varying number of slices is cropped and saved to disk as a npy file. 

**Sagittal Crops :** 3d crops for stage 2 are made by cropping a bounding box of 28% width / length of original dicom around the x/y coordinates predicted from stage 1 and using all slices. Thereby all the x/y predictions for all slices of a series are aggregated using median to increase robustness.

**Study-Level IVD model :** For a single study-level IVD the 3 view 3d-crops (Axial 3d crop, Sagittal T1 3d crop, Sagittal T2 3d crop) are loaded and reshaped to 144x288x9, 144x144x9, 144x144x9 cubes. If a view is not available it is replaced with zeros. The 2 sagittal views are concatenated on the width axis resulting in a 144x288x9 cube and then further concatenate with the axial view cube on the height axis to a single 288x288x9 3d-image. Dropping one of the views with a chance of 10% is used as additional augmentation. 
The model is of 2.5D nature and has 3 tf_efficientnetv2_s backbones where conv_pw of each InvertedResidual layer has been patched to apply 3d instead of 2d convolution. The 288x288x9 3d-image is fed to each backbone individually to predict neural_foraminal_narrowing, spinal_canal_stenosis, subarticular_stenosis separately using a CrossEntropyLoss. Additionally a loss is predicted for having a severe spinal_canal_stenosis condition in any of the 5 IVDs belonging to a study. The 4 losses are averaged and optimised as a single loss, which represents the competition metric.


## Model ensembling

Ensembling is done in the MLP of the 1st pipeline. Predicted logits of all our models are concatenated together for the three classes. Here are our final scores:

|           | SCS  | NFN  | SS   | ANY  | CV   |      |  Public LB | Private LB |
|---------|------|------|------|------|------|------|------------|------------|
| Loss  | 0.260| 0.475| 0.538| 0.255| 0.382|      | 0.355      | 0.401      |

## What did not help

- SpineNet and other medical specific models for condition level modelling. 
- Pseudo labelling on external data
- Denoising techniques
- Encoder-decoder architectures to add an auxiliary injury localization task
- Bi-encoder architectures to jointly learn on axial and sagittal images


*Thanks for reading !*
