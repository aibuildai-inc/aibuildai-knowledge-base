# 8th Place Solution & Code

Competition: rsna-2023-abdominal-trauma-detection
Rank: #8
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447706

Congratulations to all of the winners and competitors. Thank you to Kaggle and the organizers for this interesting competition.

I joined relatively late after releasing my [extravasation bounding box labels](https://www.kaggle.com/datasets/vaillant/rsna-abdominal-trauma-extravasation-bounding-boxes). My solution was pretty similar to my prior solutions in RSNA cross-sectional imaging challenges (pulmonary embolism, cervical spine fracture). I treated the task as 3 separate subtasks: predicting solid organ injury (liver, kidney, spleen), bowel injury, and extravasation. All models were based on CNN-transformer "2.5D" models. 

## Data 

I converted all DICOMs to 3-channel PNGs, where each channel was a separate CT window. I used 3 windows (soft tissue: WL=50, WW=400, liver: WL=90, WW=150, and angiography: WL=100, WW=700).

## Crop Model

Using 3D connected components, I was able to generate a mask for each CT volume in order to train a model to eliminate the empty black space. This mask was converted into bounding box coordinates for each image, which allowed me to train a 2D CNN mobilenetv3_small_050 model on 256 x 256 images to predict the coordinates. For each CT volume, I took the union of the predicted bounding boxes for each individual slice and used this to crop each individual image.

## Liver-Kidney-Spleen Organ Identification Model

Using the segmentation output from TotalSegmentator, I trained another mobilenetv3_small_050 model on 256 x 256 images to predict presence of these organs on individual slices. 

## Liver-Kidney-Spleen Injury Model

With about a week left, I decided to label some slices with injuries to the liver, kidneys, and spleen so I could train a 2D model slice-wise model. I did not release the labels since the end of the competition was near, and I did not want to create any disruption. They are now available [here](https://www.kaggle.com/datasets/vaillant/rsna-abd-trauma-organ-injury-slice-labels).

I trained a ConvNeXt-tiny model on 2D slice labels with an additional linear layer to reduce the feature dimension to 256. This included the laterality of the kidney injury (i.e., left vs. right), though I am not sure how much this actually helped in the end. The model was trained on cropped images of size 288 x 384 from cropped volumes using the above models. This model was used to extract features from each slice; CT volumes were sampled to 128 images. Thus a CT series was converted to a sequence of shape 128 x 256. 

A 3-layer transformer was trained on these sequences to predict the series-level label. A weighted binary cross-entropy loss was constructed to mimic the competition metric. The validation loss was rather unstable, so I also tracked the AUC to make sure the model was learning.

## Bowel Injury Model

I used the provided slice-wise bowel injury labels to train CNN-transformer model in the same manner as above, except I only cropped the individual images to remove black space, not the CT volume since bowel is present on more slices. Images were resized to 384 x 512. 

## Extravasation Model

Using the bounding box labels I annotated, I generated 12 nonoverlapping patches of size 128 x 128 from images of size 384 x 512 and assigned each patch with a label of injury vs. healthy. I trained a ConvNeXt-tiny model on these patches and extracted features for each patch. Thus each image was converted into a sequence of shape 12 x 256. 

I trained a 3-layer transformer using those sequences on slice-wise labels. I then used this transformer to extract features from each image of a resampled 128-slice volume, again resulting in a sequence of shape 128 x 256. This method improved performance over simply training a 2D CNN on whole images. A second-stage transformer was trained on these sequences to predict the series-level labels, using a weighted loss similar to the above.

## Inference

5-fold ensemble of the above was used for final inference. For patient-level prediction, predictions were averaged across the series, if there were 2. Softmax activation function was applied to each label group's predictions, so all the probabilities were already normalized to 1. All probabilities were then scaled by taking the square root, which improved the private LB loss by 0.4. OOF CV was 0.375.

## Additional Thoughts

I tried to incorporate 3D models, but they were taking too long to train and the performance was not as high. I was also interested in training segmentation models and training on cropped organs using the segmentation masks but did not have enough time. I tried training models on images with stacked slices as channels (i.e., each channel of the "image" was a separate slice), but this resulted in similar performance (slightly worse on LB). I tried training a single transformer on the concatenation of the features from the 3 types of models above, but this resulted in worse performance. Overall, I am happy to have won my 10th gold medal. 

Inference Notebook: https://www.kaggle.com/code/vaillant/rsna-trauma-submission-v2-1

Source Code: https://www.kaggle.com/datasets/vaillant/rsna-trauma-src
