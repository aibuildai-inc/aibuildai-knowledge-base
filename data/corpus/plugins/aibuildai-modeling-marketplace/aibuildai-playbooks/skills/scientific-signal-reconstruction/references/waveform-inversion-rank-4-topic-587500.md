# 4th place solution for the GWI competition

Competition: waveform-inversion
Rank: #4
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587500

It was a great competition. Many thanks to the Kaggle staff, the hosts, and all the active discussion and code section participants.

## Context section  
[Business context](https://www.kaggle.com/competitions/waveform-inversion).  
[Data context](https://www.kaggle.com/competitions/waveform-inversion/data).  

## 1. Overview of the Approach  
I generated a lot of additional data (probably more than 10M samples- at some point, I stopped counting). Then I trained a ~100M parameters custom ViT+2D conv model for ~3 weeks. Then I added a bit of postprocessing for the final 0.01 bump.  

### 1.1. Data generation  
I implemented Python vel-to-seis on GPU and could generate ~13k samples per hour on T4. Then I generated for many hours on Kaggle GPU (60 hours in a week is sweet!) and colab.
It's really slow compared to other top solutions, tho.
At first, I generated very heavily augmented data (cut-mix between all families+scaling+shifting, etc.). I started to train my model with this, then during the training, I refined my data generation by implementing the host method from the paper for generating CurveVel and CurveFault. I could not get the same CurveFault, but I was close. For Style, I just used cutmix between samples from the same style family. I downcast the seis to 5\*288\*70 and saved as bfloat16.  

### 1.2. model
If you are familiar with my LEAP solution, it's very similar- interlaced layers of transformers and 2D conv block (it helps a lot with convergence due to bias!). The input was casted 288\*70\*5 -> 18\*18\*384 patches, then passed through 28 blocks of 2Dconv+transformer, then casted back 18\*18\*384 -> 70\*70\*1. Optimizar was AdamW with a half-cosine lr scheduler, starting from a max lr of 1e-3.  

### 1.3. Loss  
MAE+confidence Loss (confidence Loss : MAE(confidence, MAE(targets, preds))).

### 1.4 Post-processing  
It was a small bump in the last day (only 0.01!), but the difference between 5th and 4th place. But it can be ignored in the big picture.  
#### 1.4.1  
Clipped predictions between 1500 and 4500- very very minor.  

#### 1.4.2.  
I trained a classifier on the prediction. It was a very good classifier. Then, for all families except Style A+B, I rounded to int. This, together with clipping, was 0.03 in val.  

#### 1.4.3   
Again, for all families except style, for each sample I found grouped values and changed the values of each group to the most prominent one within the group (basically binned the predictions since those families had only a small number of values in vel maps- I call it 'restoring bias'). This was 0.1 in val.  

## 2. Details of the submission  
### 2.1. Ensembling.  
At first, I started training a large model with horizontal flip augmentation + TTA. Then I realised, I can cheaply generate exact flipped samples by only recalculating the central seis out of 5- the rest are flip-identical. But for the middle, the source has one pixel shift after flipping. Training on exact flipping was better than flip-augmenting, so I continued my training with two versions- one with no flipping and the second with everything flipped (to use the flip-augmentation data the model already learned). I ensembled these two models for 8.7 LB.  
In the last week, I noticed the models were a bit weaker on Style B- during training, I focused more on Curve families. So I decided to train an additional, third model in the last week, focused on Style- it was a success and achieved a bump of 2 for StyleB. Then I ensembled it with the other two, using the classifier model for calculating different weights for each family based on my validation split (10K samples in total, 1K for each family). This gave me another 0.1 bump to 0.86 (and the final 0.1 was from post processing, as I mentioned).  
Also, I ensembled several checkpoints for each model (about 10 checkpints give or take).  

### 2.2 Hardware
I used Kaggle and Colab T4 for data generation and TPU for training. Spent about 400$ in total on colab credits.  

## 3. Sources
[MaxViT implementation in tensorflow](https://github.com/tensorflow/models/tree/f007603b50b4db38907594a156994a4e983d2d31/official/projects/maxvit)- I based my 2D conv block on their implementation with minor modifications.  

## Code
[Here](https://github.com/shlomoron/GWI-solution).
