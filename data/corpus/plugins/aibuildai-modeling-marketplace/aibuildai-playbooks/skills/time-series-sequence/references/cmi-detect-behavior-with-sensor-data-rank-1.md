# 1st place solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #1
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution

**Links to code:**
- [Training code on github](https://github.com/statist-bhfz/kaggle_cmi_1st_place_solution)
- [Inference notebook](https://www.kaggle.com/code/yuanzhezhou/final-sub-cmi3)

## Solution Writeup

Congratulations to all the winners and a big respect to the organizers for putting together such a challenging and rewarding competition. 
Special thanks to my teammates [yuanzhe zhou](https://www.kaggle.com/yuanzhezhou) and [Devin Anzelmo](https://www.kaggle.com/devinanzelmo) for the great collaboration and team spirit.

## Solution Overview

-   Large ensembles for IMU-only and all-features samples, assembled as weighted sum of logits

-   Post-processing to take into account limited number of samples with each class and with each orientation per subject (already described [here](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/603544))

-   Additional models for orientation prediction

------------------------------------------------------------------------

## Devin's part

**1. Preprocessing/feature engineering:**

- Removed subjects who wore device incorrectly from both training and validation

- For TOF subsample by taking nine 2x2 squares and calculating mean for each (turn -1 to null first)

- Some models trained with temporal difference of above TOF features and thermal

- IMU: raw 7 + temporal diff of `acc`, rotational diff as posted in notebooks, as well as magnitude of `acc`/`rot`; 16 total

- preprocess left handed to match right handed for IMU and THM/TOF. Flip `acc_x`, convert `rot` to xyz euler and flip sign for y and z

- robust scaler 

**2. Augmentation:**

- up scaling/down scaling all features 

- random noise 

**3. Models:**

- CNN GRU with different stems for different features. Feature extractor is four branches of CNN with no pooling

- Pure CNN. Same feature extractor as above but three blocks of residual at end with expanding kernal (5, 9, 13)

- dense GRU, scores lower but helps a little in ensemble

- some self excitation models, some not combining helps

- 3D CNN for TOF data scores about 0.83 but blends well

**4. Training:**

- batch size 64

- train for 60-80 (depending on model) iters with AdamW `lr = 0.0005`

**5. Ensemble:**

- blend IMU models with all-features models 

- blend TOF-only model improves all data score

------------------------------------------------------------------------

## Ogurtsov's part

**1. Filtering data:**	
	
- [all models] remove sequence SEQ_011975 without gesture	
	
- [all models] remove sequences with gesture/total ratio < 0.2 (this subset includes almost all shortest sequences)	
	
- [models with THM and TOF data] remove sequences with TOF completely missing	(missing values patterns in TOF - 0%, 20% or 100%; 20% missing is ok)	

- few models were trained without "problematic" subjects SUBJ_045235 and SUBJ_019262
	
**2. Preprocessing:**
	
- fill quaternion missing values, if possible	
	
- mirror data for left-handed subjects 

- replace -1 TOF values with 500 because -1 encodes large distance	
	
- ultimate filling NaNs with `seq.ffill().bfill().fillna(0)`
	
- pad/crop by 94 percentile in each fold (~120 max seq length)	
	
- no scalers - it gave the largest improvement at early stage	
	
**3. Feature engineering:**	
	
- nothing for THM and TOF	
	
- 35 IMU features:	
```
['acc_x', 'acc_y', 'acc_z', 'acc_mag', 'acc_mag_jerk', 'jerk_x', 'jerk_y', 'jerk_z', 'jerk_magnitude', 'acc_xy_corr', 'acc_xz_corr', 'acc_yz_corr',	
rot_w', 'rot_x', 'rot_y', 'rot_z', 'rot_angle', 'rot_angle_vel', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_vel_magnitude', 'angular_distance',	
acc_x2', 'acc_y2', 'acc_z2', 'acc_mag2', 'acc_mag_jerk2', 'jerk_x2', 'jerk_y2', 'jerk_z2', 'jerk_magnitude2', 'acc_xy_corr2', 'acc_xz_corr2', 'acc_yz_corr2']
```

`acc_x2` and other `***2` features are important part of the solution. 
These features are calculated on values transformed by `remove_gravity_from_acc()`.	
	
**4. Models:**	
	
- bunch of models with different architectures for training on IMU-only data and similar models for training with all features	
	
- main differences in time dimension processing: LSTM-, attention- or CNN-based	
	
- key part - independent processing groups of features at first layers	
	
**5. Augmentations:**	
	
- mixup with `MIXUP_ALPHA = 0.4`
	
- augmentation from [CMI | Sequence Data Augmentation](https://www.kaggle.com/code/alejopaullier/cmi-sequence-data-augmentation) (thanks [moth](https://www.kaggle.com/alejopaullier)):	
```
transforms = Compose([	
    TimeShift(p = 0.35, padding_mode = "zero", max_shift_pct = 0.25),	
    TimeStretch(p = 0.35, max_rate = 1.5, min_rate = 0.5),	
])
```
with some variations.

It was hard to pick up right augmetations with right probability & strength and apply it in the correct order.	
	
**6. Training:**	
	
- CV5 with group by subject K-fold, 5 models are used on inference	
	
- 200 or 220 epochs, patience 30 or 35	
	
- `torch.optim.Adam` with `lr = 4e-4` or `lr = 5e-4` and `weight_decay = 3e-4`	
	
- `CosineAnnealingWarmRestarts`; `LinearLR` + `CosineAnnealingWarmRestarts` for attention-based models	
	
- one model was trained with focal loss and class weights	(class weights obviously mean nothing for mixup samples, maybe it can serve as regularization)	

- one model contains bug in cross-attention (shared weights for 3 different attentions) but works pretty well

**7. Some tricks:**	
	
- pick best result for each fold among 3 runs

- increase sequence length by 5 on inference (it doesn't improve CV score but effectively decorrelates models)

------------------------------------------------------------------------

## zyz's part

1.1. Using the pipeline structure from [CMI3 Pyroch Baseline+model_add, Aug, folds](https://www.kaggle.com/code/myso1987/cmi3-pyroch-baseline-model-add-aug-folds?scriptVersionId=248310337)

1.2. Fixing many bugs improved my score to 0.85.

2. Models: simple combinations of RNN and CNN1D.

3. Feature engineering using pytorch, which avoids problem of consistency but makes the training slower.

4. Basic preprocessing mentioned by most teams: fillnan, handedness, etc.

5. Post-processing based on data structure.

6. Augmentation idea from teammates:

6.1. Mask data during training helps.
