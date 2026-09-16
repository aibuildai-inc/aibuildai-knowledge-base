# 15th Place Solution(PB 0.173 Solution)

Competition: isic-2024-challenge
Rank: #15
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532580

Hello, this is Youhei Tomio.
I would like to thank the hosts for organizing the competition.

I was able to achieve a ranking of 15th with the PB 171 notebook, but I also had a notebook for PB 173.There were hardly any differences in approach. The main differences were the type of image model used from timm and whether there was target leakage when incorporating predictions from the image model into the tabular model. In the PB 171 solution, there was target leakage, but in PB 173 solution, there was no target leakage. Here, I will post the solution from PB 173 solution, which offers a superior approach.
[image]

## Summary of the solution
[image]




## Image Model
I selected six models from timm and incorporated their predictions into the tabular data. 
Data augmentation and TTA were only added as they were in the public notebook.

| Model                                               | CV   | Input Size |
|--------------------------------------------------------------|---------|------------|
| tf_efficientnet_b0_ns                                         | 0.1545  | 384        |
| tf_efficientnetv2_m.in21k                                     | 0.1455  | 224        |
| coatnet_rmlp_2_rw_224.sw_in1k                                 | 0.1555  | 224        |
| eva02_small_patch14_336.mim_in22k_ft_in1k                     | 0.1602  | 336        |
| convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320        | 0.1644  | 224        |
| swin_large_patch4_window7_224.ms_in22k_ft_in1k                | 0.1594  | 224        |

I used [Triple Stratified Leak-Free KFold CV](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/165526)



## Tabular Model
I used ~~four~~ 3 models: LightGBM, CatBoost, XGBoost,~~and HistGradientBoost~~. I also applied oversampling and undersampling , using the rates provided by @greysky for all the models as they were.

## Feature Engineering
I utilized the "ugly duckling feature" posted by @richolson . However, I did not include features containing information from image data in the LightGBM model, but incorporated them into the other~~ three~~ 2models. The LightGBM model tended to overfit on image data.

I selected several columns from the "ugly duckling feature" posted by @richolson and created unsupervised learning-based "ugly duckling features" using the HBOS algorithm from the  [pyod library](https://pyod.readthedocs.io/en/latest/index.html).
This HBOS approach improved the LB score by +0.001.


Here is a histogram showing the difference in the distribution of HBOS based on the target .
[image]


## Validation Strategy 

I used [Triple Stratified Leak-Free KFold CV](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/165526) recommended in the previous competition. However, since the correlation between LB and CV was low, I didn’t trust the CV scores. Therefore, I primarily relied on the LB scores. Additionally, I periodically checked the robustness of the model by changing the seed value after each update I made and observing how the LB scores changed.


[Submit code LB0.173](https://www.kaggle.com/code/youheitomio/submit-lb0-173-pb0-186)
**
I won the gold medal through luck. I would like to thank all the Kagglers who shared their various solutions. Thank you very much!
