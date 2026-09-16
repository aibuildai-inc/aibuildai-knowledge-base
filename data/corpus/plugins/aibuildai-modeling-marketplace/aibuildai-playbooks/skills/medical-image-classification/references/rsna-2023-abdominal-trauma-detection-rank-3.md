# 3rd Place Solution

Competition: rsna-2023-abdominal-trauma-detection
Rank: #3
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447464

I salute you all for fighting to the end. Thanks also to the RSNA for their support. It was another great competition.

## Overview
1, The 3D segmentation was trained with the given masks and each organ was cut into a cube shape using the predicted results.
2, Multiple organ cubes were each entered into various 2.5D+3D classification models and the results enumerated.

## Segmentation
I used Qishen's 3D segmentation code. https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607
The models used were resnet18 and resnet50. The average of the output of all models was used as a mask.

## Crop
Since information around the organs is essential for trauma detection, the mask was slightly enlarged before the boxes were cut out. Two patterns of mask sizes were employed and two datasets were created for each organ.

## Classification

All classification models follow a 2.5D + 3D structure. Typically, multiple 2.5D images are generated from the organ box and input into the model. The input sizes are (8, 15, 3, 128, 128), where 8 is batch_size, 15 is image, and 3 is channel. Each image is transformed into a feature map through a 2D CNN and input to subsequent processes such as pooling and lstm.
Because of the correlation in injuries between organs, the problem was solved in a multiclass problem, using essentially all targets. I trained a variety of model patterns, including:

・Multiple Image Sizes
・Multiple image counts
・Multiple necks (average pooling / max pooling / lstm / gru)
・Multiple crop sizes
・Multiple backbones (convnext / se_resnext / maxvit / caformer / xcit)
・Multiple augmentation sets
・Multiple epochs(without early stop)
・Multiple targets (all targets / single organ targets)
・Some models are pre-trained with image-level bowel / extravasation labels and used as initial values for weights.
・Some models reduce noise by using box*mask as input. There is also a model that uses box*mask as input to reduce noise. This type of model is specific to the liver. Because of its shape, the liver has a lot of noise information if you just crop it with a box, so masking was very effective.
・Some models use all organ boxes for training. Different image sizes/number of images are used for each organ to ensure the same resolution. A custom sampler was defined so that only boxes of the same organ exist in the same batch, allowing simultaneous training with different sizes/numbers.

The following are those that have made a particularly significant contribution to accuracy:
・masking for liver model
・custom sampler for all class models
・2types of crops

## Ensemble
A simple weighted average was performed for each target.
Below is a simplified weights.

```
{
'bowel_injury':
    {
        'liver_gru_chaug_256_cropv1': 0.0,
        'liver_maxvit_224_cropv1': 0.0,
        'liver_maxvit_224_cropv2': 0.2,
        'spleen_gru_128_cropv1': 0.0,
        'spleen_maxvit_224_25epochs_cropv1': 0.0,
        'kidney_maxvit_224_cropv1': 0.0,
        'kidney_caformer_192_cropv2_pretrain': 0.2,
        'kidney_maxvit_224_cropv2': 0.0,
        'kidney_maxvit_224_25epochs_cropv2': 0.0,
        'bowel_lstm_256_n15_cropv1': 0.0,
        'bowel_288_n25_cropv1_pretrain': 0.3,
        'bowel_288_n25_25epochs_cropv1_pretrain': 0.2,
        'all_pretrain_cropv1_input_bowel': 0.1,
        'all_pretrain_cropv1_input_kidney': 0.0,
        'all_lstm_pretrain_cropv2_input_kidney': 0.0,
    },
'kidney_healthy':
    {
        'liver_gru_chaug_256_cropv1': 0.1,
        'liver_maxvit_224_cropv1': 0.0,
        'liver_maxvit_224_cropv2': 0.0,
        'spleen_gru_128_cropv1': 0.0,
        'spleen_maxvit_224_25epochs_cropv1': 0.0,
        'kidney_maxvit_224_cropv1': 0.2,
        'kidney_caformer_192_cropv2_pretrain': 0.1,
        'kidney_maxvit_224_cropv2': 0.2,
        'kidney_maxvit_224_25epochs_cropv2': 0.2,
        'bowel_lstm_256_n15_cropv1': 0.0,
        'bowel_288_n25_cropv1_pretrain': 0.0,
        'bowel_288_n25_25epochs_cropv1_pretrain': 0.0,
        'all_pretrain_cropv1_input_bowel': 0.0,
        'all_pretrain_cropv1_input_kidney': 0.1,
        'all_lstm_pretrain_cropv2_input_kidney': 0.1,
    },
}
```

The final oof log loss is as follows:
```
{
'bowel_injury': 0.0592,
'extravasation_injury': 0.2004,
'kidney_healthy': 0.1101,
'kidney_low': 0.0958,
'kidney_high': 0.0451,
'liver_healthy': 0.189,
'liver_low': 0.1984,
'liver_high': 0.0346,
'spleen_healthy': 0.1746,
'spleen_low': 0.1651,
'spleen_high': 0.0788
}  

```
## Post processing
After a simple weighted average of each model, each prediction was further weighted. This is the same post-processing as in the public notebook.
I also tried stacking which directly optimizes the metric, but the results were slightly worse than the simple pp due to overfit.
The final CV is 0.3316 and the respective scores are as follows.

|           | bowel   | extravasation | kidney | liver  | spleen | any    |
|-----------|---------|---------------|--------|--------|--------|--------|
|  | 0.095  | 0.4853   | 0.2434 | 0.3489 | 0.3751 | 0.4417  |  

<br>

train  code: https://github.com/yujiariyasu/rsna_2023_abdominal_trauma_detection
inference code: https://www.kaggle.com/code/yujiariyasu/3rd-place-inf-code/notebook

Thanks everyone for your hard work!
