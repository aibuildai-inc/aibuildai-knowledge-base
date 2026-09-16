# 10th Place Solution (monnu part)

Competition: waveform-inversion
Rank: #10
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587412

Congratulations to everyone who organized this event and to the winners! I am deeply thankful to my teammate @ren4yu for working in this challenging competition. Additionally, I would like to express my gratitude to kagglers for sharing valuable knowledge.

The final submission is a weighted average of contributions from two members. This is my contribution. You can find ren4yu's excellent solution [here](https://www.kaggle.com/competitions/waveform-inversion/discussion/587429).

## Solution Overview

- Based on Berthly's ConvNeXt model
- Fine-tuning by applying the forward model to test data
- Post-processing with backpropagation

## 1. Model Structure

- Base: Berthly's ConvNeXt UNet
- Changes:
  - Resized images with interpolation
  - Backbone: ConvNeXt Small → ConvNeXt Large

## 2. Training Process

After initial training, we created pseudo-labels using the forward model on test data, then fine-tuned to improve generalization.

### Stage 1: Initial Training

- Trained on the OpenFWI dataset
- Generated velocity maps from test data
- Converted velocity maps to input images with FWM
- Added these as "pseudo-labels" for the next stage

### Stage 2 and Beyond: Data Augmentation & Fine-Tuning

- Added Stage 1's test data pairs to the original training data
- Fine-tuned with the augmented dataset
- Repeated this cycle 3 times

## 3. Post-Processing on Test Data

This part was a contribution from my teammate yu4u, and it worked well with my model too. We used FWM and backpropagation to optimize the L1 loss between input data and reconstructed predictions for each test sample. For more details, please refer to my teammate's explanation.

## 4. Results (CV and LB)

| Stage                        | CV   | LB   |
|------------------------------|------|------|
| Initial Training             | 23.9 | 27.2 |
| Fine-Tune (3round)                | 16.8 | 17.2 |
| Post-Processing on Test Data | 12.64 (1/100 downsample) | 13.5 |


The final submission was blended with models from other team members to obtain the final score.
