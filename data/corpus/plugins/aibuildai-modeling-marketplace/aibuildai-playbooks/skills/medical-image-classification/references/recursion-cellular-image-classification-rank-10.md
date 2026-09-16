# 10th place - Pure Magic Solution

Competition: recursion-cellular-image-classification
Rank: #10
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110375

Congrats to all the winners and thanks to the host and Kaggle hosted such an interesting competition. Huge thank to my teammates you are the greatest!

**Overview:**
All our models were trained with 512x512x6 images. We sampled a random site for each sample per epoch. All our were 2-headed.
First head: classification head had an output with 1139 neurons, 
Second head: embeddings.

**Challenges:**
Validation 
Specific experiments like U2OS-4 failed 

**Validation for CNNs:**
We noticed that specific experiment types perform very differently from others. We chose the hardest experiment for our model and built the validation based on it. 

**Training Augmentations:**
- flips (horizontal, vertical)
- rot90 
- transpose
- shift (0.25) with 101 reflections 
- rot360 (-180, 180)
- cutout (32x32, 2 holes)
- noise (gaussian, localvar, poisson, salt&amp;pepper, speckle)
- clahe
- gamma (0.9,1.1)

**Data Pre-Processing:**
per channel (img - img.mean()) / (img.std() + 1e-6)
We noticed that due to heavy augmentations we had a true_division warning. This warning caused batchnorms to feel bad, so we did this trick to prevent zero division.

**Model training:**
We train our models with NVidia Apex and Pytorch 1.2 on 8x1080ti GPU server for 110 epochs. The training takes 1-3 days depending on the model.
We did not use oversample.
Because of the long iteration, we didn’t use any K-Fold training. We wanted to train it closer to the end of the competition, but we didn’t have much time. 

**Optimizer:** SGD

**Scheduler:** init LR 0.1, 
5 epochs warmup,
drop LR every 40 epochs with factor 0.1

**Loss Functions:**
Smooth CrossEntropy for classification, Center Loss for embeddings. 

**Model:**
Our final model is Senet154 trained on train data + pseudo-labels.
To produce pseudo-labels were using an ensemble SeNet154, SeResnext50, SeResnext101, Polynet, EfficientNet-b6, ResNeXt101-wsl. 

**Test time augmentations:**
output = (model(img1) + model(img2) + model(img1.flip(2)) + model(img1.flip(3)) + model(img2.flip(2)) + model(img2.flip(3)) / 6.

**Post-processing:**
Hungry experiment re-calibration and plate re-calibration
https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73803#latest-438270 

**What didn’t work or had about the same performance:**
- Mix-up, manifold mixup
- ArcFace
- Last stride 1
- Focal Loss
- XGBoost ensembling 
- Metric search based on the embeddings distances
- GeM
- Different optimizers like Adam, Radam, Ranger, etc.
- ResNeXt101-wsl performed worse than expected.
