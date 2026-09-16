# 4th Place Solution

Competition: rsna-2023-abdominal-trauma-detection
Rank: #4
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447848

Thank you to RSNA and Kaggle for hosting this competition. Congratulations to all the winners and participants.

## Overview
I employed a 2.5 pipeline and trained for both classification and segmentation tasks.

## Dataset
I utilized 3D masks from TotalSegmentator and subsequently retrained a 2D model for the liver, spleen, bowel, kidney, and body. The DICOM images were rescaled to (1, 1, 5) and stored with a 5-channel mask.
For each epoch, I first sampled N=14 frames from the rescaled array, then used the body mask to filter out hands or other irrelevant areas.
I used the organ mask to limit the Z-axis space, as slices without the target organ might contain less valuable information.
## Model
I used a Unet model integrated with Pyramid Vision Transformer V2 and MaxViT encoder. Transformers outperformed the convolutional models, especially for the extravasation target.

Pretraining on the 2D mask facilitated convergence.
6 classification heads were employed for prediction.
Perhaps the lack of an RNN layer is the primary reason I didn't match the performance of the top teams.
## Loss
I used the CE loss with weights identical to the metric.

## Results
| Encoder | CV | LB |
| --- | --- | --- |
| pvt-b2 | 0.3783 | 0.41 |
| pvt-b3 | 0.3750 | 0.41 |
| pvt-b4 | 0.3786 | 0.4 |
| maxvit_t | 0.3810 | 0.42 |
| ensemble | 0.3570 | 0.4 |
| ensemble w/ scale | 0.3530 | 0.39 |
## Code
[Github link](https://github.com/iseekwonderful/RSNA-2023-Abdominal-Trauma-Detection-4th-Place-Code.git)
