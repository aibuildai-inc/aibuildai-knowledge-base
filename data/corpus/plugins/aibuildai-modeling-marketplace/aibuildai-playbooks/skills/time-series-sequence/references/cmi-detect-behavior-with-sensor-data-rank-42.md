# 42th Solution (Late sub private=0.854)

Competition: cmi-detect-behavior-with-sensor-data
Rank: #42
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/42th-solution-late-sub-private0-854

## TL; DR

- Converted left-handed sensor data into pseudo right-handed data, which (I believe) result in not shaken much in Private LB
- Triple tower model which consists of IMU/ToF/IMU+ToF branch
- Multi modal training is slightly stable: gesture, orientation, target-type, behavior
- Key factor: splitting temporal attention module per each target is key (I hypothesize that different time frames are important for each targets)

## Architecture



- earlier model has squeezeformer blocks before temporal attention modules, but later these blocks are discard because only using several layers of temporal attention module performs equivalently.
- developed architecture which replaced temporal attention module with CrossAttention, BERT, LSTM. Among these, BERT is the best, but LSTM model seems to play better orthogonality in ensemble

## Result

### Single Architecture (5-fold)

|model_id|model|#models|augmentation|ensemble_method|flip correction|CV (IMU-only, Full)|Public LB|Private LB|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|1|d01_sf_mca_sp_triple_tower_v4_1_l3_b3_cdf_stm256|5|heavy12|prob|-|0.825 (0.787, 0.863)|-|-|
|2|d01_sf_mca_sp_triple_tower_v4_1_l3_b3_cdf_stm256|5|heavy13|prob|-|0.839 (0.811, 0.867)|0.842|0.832|
|3|d05_sf_bert_isolated_ncl_full_l0_b6_cdf_stm128|5|heavy14|prob|-|0.843 (0.809, 0.878)|0.845|0.840|
|3|d05_sf_bert_isolated_ncl_full_l0_b6_cdf_stm128|5|heavy14|prob|✅|0.843 (0.809, 0.878)|0.845|0.846|
|4|d09_bilstm_isolated_ncl_full_b1|5|heavy15|prob|-|0.845 (0.814, 0.875)|0.841|0.832|
|5|d09_3b_bert_null_emb_isolated_ncl_full_b12|5|heavy15|prob|-|0.849 (0.812, 0.885)|0.838|0.833|

- `l{n}_b{m}` means, n-layer of squeezeformer and m-layer of temporal attention module

### Ensemble

ensemble_id|model|#models|ensemble method|flip-correction|CV (IMU-only, Full)|Public LB|Private LB|Note
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
top3|ensemble {3,4,5}|15|prob|-|0.858 (0.826, 0.890)|0.853|0.841|Final Sub
top4|ensemble {2,3,4,5}|20|prob|-|0.861 (0.83, 0.892)|0.853|0.843|Final Sub
top4|ensemble {2,3,4,5}|20|rank|-|0.859 (0.828, 0.89)|0.852|0.844|Late Sub
top4|ensemble {2,3,4,5}|20|prob|✅|0.861 (0.83, 0.892)|0.853|**0.854**|Late Sub

## Convert left-handed sensor data into pseudo right-handed sensor data

- converted left-handed data into "mimic" right-handed data.
- I didn't tested on LB, but always perform better than raw data in CV



Concept and hand-written note is already shared with:
https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/603566

Almost the same approach that Jack's does, so details are omitted here.
https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/6th-place-solution

## Data Correction of Flipped Subject in Train set

Same as above.

## Data Correction of Flipped Subject in Private Test set [Late Submission]

- automatically generate flipped data by applying inverse transform of Flipped subject with p=0.066
- train classifier 100 epochs with the similar architecture of gesture classification model

**Result**

- precision: 0.996
- recall:  0.980

Data correction if prob > 0.5 -> Private LB is improved by +0.01 (0.844 -> 0.854), which is significant impact.



## Other Training Details

- cosine schedule which restarts **every epoch** -> slightly stable compared to one-cycle cosine schedule 
    - Note that this schedule is considered as kind of "schedule-less" training where we can restart training at any epochs if checkpoints are available



- Data Augmentation
   - replace non-gesture frames sampled with different sequence with the same orientation/gesture -> CV is slightly improved
   - resampling only on gesture frame
   - Rotate around Z axis (-90, 90)
   - other augmentations stabilize training, but minor improve on CV

```python
    "heavy15": Compose(
        # Resample All Series
        RandomCropOnlyGesture(max_crop_ratio=0.5, min_gesture_length=10, p=1.0),
        Choice(
            RandomResampleOnlyGestureV2(resample_rate_range=(1.0, 2.0), p=0.4),
            RandomResampleOnlyGestureV2(resample_rate_range=(0.5, 1.0), p=0.2),
            ResampleOnlyGesture(sample_rate=(1.0, 1.2), p=0.2),
            p=0.8,
        ),
        # IMU Transformations
        Compose(
            RandomRotateZ(angle_range=90, p=1.0),
            RandomEulerRotation(
                angle_range=20,
                transform_features=("acc",),
                p=1.0,
            ),
            Choice(
                RandomEye(["sensor_basis_world"]),
                RandomZeroOut(["acc"]),
                p=0.15,
            ),
        ),
        # ToF/Thermo Transformations
        Compose(
            RandomAffine(
                keys=("tof",),
                translate_range=(-1, 1),
                rotation_range=(-15, 15),
                scale_range=(0.9, 1.1),
                p=0.3,
            ),
            RandomAdditiveNoise(key="thermo", shift=1.0, scale=0.3, p=0.3),
            RandomAdditiveNoise(key="tof", shift=5.0, scale=10.0, p=0.3),
            SensorChannelDropout(key="tof", drop_prob=0.5, p=0.5),
            SensorChannelDropout(key="thermo", drop_prob=0.5, p=0.5),
            Choice(
                RandomZeroOut(["thermo"]),
                RandomZeroOut(["tof"]),
                p=0.25,
            ),
        ),
        Choice(
            # Completely Drop ToF/Thermo or IMU data
            RandomZeroOut(["tof", "thermo"], p=0.7),
            Compose(RandomEye(["sensor_basis_world"]), RandomZeroOut(["acc"]), p=0.05),
            Identity(p=0.25),
        ),
    ),
```

## Rererence

- [Ensemble Top4 - Private=0.854] https://www.kaggle.com/code/tatamikenn/top4-flip-detection/notebook?scriptVersionId=260191735
