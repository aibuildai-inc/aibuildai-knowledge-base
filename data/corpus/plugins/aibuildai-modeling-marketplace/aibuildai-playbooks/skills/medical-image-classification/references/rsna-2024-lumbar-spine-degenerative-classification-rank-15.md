# 15th Place Solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #15
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539981

Congrats to all winners and those seeing themselves better data scientists than who they were before completing this competition! Despite being 2 ranks short of gold, we find this competition really interesting as there is no trivial way to approach this competition, which makes it much more interesting.

Most importantly I'd like to thank @viktorcikojevic for teaming with me on this (any loads of past) competition. Without him, there's no chance I'd have come this far.

## Overview

On a higher level, our pipeline is depicted in the following diagram:



It consists of three stages:
1. Keypoint detection stage
1. Crop proposal stage
1. Crop classification stage

### a brief word on what motivated this design choice:

- This pipeline should allow all models to train at a *per-image* level, rather than *per-patient* level. There is only about ~2000 patients, so we thought this would be a better way to utilize all of the data and prevent overfitting. 
- A lot of information are inferrable between each model's output, which allowed us to include a helpful bias to the model. For example, we know T2 runs right at the center of the person, so all T1 keypoints on their left are left T1 keypoints, same for the right hand side. This means we can take some shortcuts on what the model must learn.

##### When it comes to implementation, it means we did the following:
- At the keypoint detection stage, our T1 keypoint model will only predict 5 classes: L1/L2, L2/L3, L3/L4, L4/L5, L5/S1 **and not 10**. No sides are predicted at this stage.
- At the keypoint detection stage, our axial keypoint model will only predict 2 classes: left and right keypoint, **not 10**. No levels are predicted
- the crop classifier's job is to take a crop in, and output 3 logits - mild / moderate / severe - it doesn't predict the condition
- the Crop proposal stage does 3 main jobs
  - fill in the sides of each T1 keypoint (because the keypoint model only knows the levels)
  - fill in the level of each Axial keypoint (because the keypoint model only knows the sides)
  - aggregate per-image predictions into the final 25 keypoints for each patient.


In the sections below we will describe each stage in more detail.


## Keypoint Detection Stage

Here we developed a segmentation model that outputs a map of keypoints for each image. For each of the conditions, we train a SMP (Segmentation Model Pytorch) model with `timm` backbone. The model takes the 3 consecutive `instance_number` channels as inputs and outputs a multi-channel heatmaps.

### T2 Keypoint models
For T2 models, we just use the dataset shared by @brendanartley here https://www.kaggle.com/datasets/brendanartley/lumbar-coordinate-pretraining-dataset.

With us joining when there's on month remaining, we did not find ways to exploit the left keypoints, so we just train with the right keypoints only.

### T1 & Axial Keypoint models
For T1 and Axial keypoint models, we just trust the labelled keypoints as-is and trained our model using those. Mostly because we're lazy (to remove all the labelling noise), and also we're a little short on time

One reason we think we can afford some labelling noise is the fact that we took shortcuts to minimise what labels the models are trained on, i.e. the T1 model doesn't care about the sides of the keypoint, so we're robust to side flips, and the Axial model doesn't care about the level, so we're robust to any noise wrt. level labels.

## Crop Proposal Heuristic

We use heuristics to go from output keypoints to the final 25 keypoints for the patient. It needs to accomplish 3 things:

- fill in the sides of each T1 keypoint (because the T1 keypoint model only knows the levels)
- fill in the level of each Axial keypoint (because Axial keypoint model only knows the sides)
- aggregate per-image predictions into the final 25 keypoints for each patient.

The overview of the Crop proposal heuristics is shown here:



### Step 1: Argmax T2 Instance Number

This step is rather simple: for each of the 5 T2 level, we find the instance number with the highest confidence from the keypoint model. After this step ends, we have 5 T2 keypoints for the patient.

### Step 1: Infer T1 Sides

Because we know the T2 keypoints from the previous step, we can now compute the XYZ position in the world coordinate using the dicom's metadata.
Doing so will tell us the XYZ coordinates of the spine, where the X axis points from the right hand of the patient to the left. So now inferring the sides
is rather trivial:
> for a given T1 keypoint on level L1/L2, if it's X value is higher than its T2 on level L1/L2, then it's a left keypoint, otherwise it's the right keypoint.

This operation has an accuracy of 97% on determining the side of each T1 keypoint, the remaining 3% is either the T1 keypoint not being detected at all, or labelling noise.

There are some edge case that needs handling, which turns this step's logic into
> for a given T1 keypoint on level L1/L2, if it's X value is higher than its T2 on level L1/L2, then it's a left keypoint, otherwise it's the right keypoint
> 
> but if T2 on level L1/L2 is not detected, then use T2 on level L2/L3 instead, if that's missing too then use t2 on L3/L4, and so on

### Step 3: Argmax T1 instance number

Pretty much step 1 applied to T1 keypoints - for each side and level, we find which `instance_number` has the highest confidence. 
This step gives us 10 final T1 keypoints for the patient.

### Step 4: Guess Missing T1 Keypoint

Under any incident of T1 keypoints missing, if its "twin" exists, then just mirror it over and call it a day

> if T1 left L1/L2 went missing, mirror T1 right L1/L2 around T2 L1/L2, and blindly claim that's the T1 left L1/L2 location

This step gives minor improvements to the cv (around 0.001 to 0.002)

### Step 5: Infer Axial Levels

Because we know T2 keypoints, each have their levels. Then we can use that to guess what level each axial keypoint should have.

> for a given Axial keypoint, look for its closest T2 keypoint using its XYZ coordinates, and take that T2 point's level as the Axial point's level

This step is vulnerable to missing T2 keypoints (e.g. if T2's L2/L3 is missing, no Axial keypoint can correctly have L2/L3 as level). To combat this issue, 
we just linearly interpolate all missing T2 keypoints before inferring axial levels.

This operation also has an accuracy of 97% on levels of each Axial keypoint, the remaining 3% is mostly driven from the T2 keypoints being wrong, which affects the level lookups, or the axial keypoints being missing altogether. 

### Step 6: Argmax Axial Instance Number

Pretty much step 1, but now applied to Axial keypoints - for each side and level, we find which instance number has the highest confidence. 
This step gives us 10 final Axial keypoints for the patient.

### Step 7: Guess Missing Axial Keypoints

This step sadly doesn't exist for us. I tried many approach in imputing these missing axial keypoints, but none of them really work out that well. 
One reason is that unlike T1 keypoints where normally only one point goes missing and mirroring fixes the issue, the Axial keypoints of the same level normally go missing together.

All experiments here leads to worse cv score, so we just accepted the fact that we just can't do this and miss keypoints..


## Crop Classification Stage

At the end of our pipeline is a classification model is a 9-class model that's construced as follows:



It's a `timm` backbone and 3 heads, one for each series type. Each head outputs 3 classes - the severity of the given image *if* the image is from that series type. For example, if an image is an axial image, the forward pass sends it though red path of timm backbone + axial head, and only the last 3 entries (the subarticular severities) are filled, the T1 and T2 logits are filled with -100 so that the probabilities are 0 after softmax.

---

Thank you very much for reading this far. We hope this has been informational, or at least entertaining :)
