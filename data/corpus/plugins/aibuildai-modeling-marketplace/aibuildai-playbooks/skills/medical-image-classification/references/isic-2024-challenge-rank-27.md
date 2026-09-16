# 27th Place Solution

Competition: isic-2024-challenge
Rank: #27
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532620

First of all thanks to the kaggle community, the organisers for holding such a great competition!

This is my first time writing a post-game solution, so please bear with me if I'm not good at it!

My model is using the VotingClassifier to vote on the predictions of lgb+cb+xgb. The features are both tabular data and pseudo-labelling of image predictions using ImageNet.

# Tabular Data
Since I'm not very good at tuning gbdt models, the construction of the tabular data features referenced public notebooks: [https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data)

# ImageNet
ImageNet references the code of the SIIM-ISIC 2020 winner: [https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution).
 I have trained 4 models which are tf_efficientnet_b4_ns, tf_efficientnet_b5_ns, tf_efficientnet_b6_ns and tf_efficientnet_b7_ns. I have also tried other models like resnet101, seresnext etc. but none of them gave better results on cv than efficientnet. The training details of the model are given below:
| Model Name | Training Data |
| :---: | :---: |
| tf_efficientnet_b4_ns | ISIC 2024 Image + Meta Data |
| tf_efficientnet_b5_ns | ISIC 2024 Image + ISIC 2020 Image + ISIC 2019 Image |
| tf_efficientnet_b6_ns | ISIC 2024 Image + ISIC 2020 Image + ISIC 2019 Image |
| tf_efficientnet_b7_ns | ISIC 2024 Image + ISIC 2020 Image + ISIC 2019 Image |

The initial learning rate for all models was 3e-5 and batch_size was 128 using Adam optimiser and GradualWarmupScheduler. All models use BCEWithLogitsLoss and 5-fold cross-validation. The resolution of the image data was 256 * 256. 

##  About why I trained the model this way
On the one hand, I looked at the ImageNet model used by public notebooks, whose pseudo-labels used to give training to the gbdt model I thought were at serious risk of data leakage, so I re-trained my own CNN model.
On the other hand, I believe that models with larger parameter counts are better able to capture the details corresponding to positive samples, so efficientnet_b5 ~ b7 instead of using the meta data from this competition, I added data from 2020 and 2019 to alleviate the category imbalance while enlarging the training samples.
When generating training pseudo-labels, the model for each fold only generates labels corresponding to the validation set to prevent data leakage. The oof strategy is used when generating test pseudo-labels.

To ensure that I could run it in less than 12h, I turned on half-precision and used two GPUs during CNN network inference.

I ended up with a cv score of around 0.1814 and a public LB score of 0.184.

# Finally
Since I entered two weeks before the end of the competition, I tried to find a team at first, but no one paid any attention to me, so I had to go on alone. May every participant not only win honours on the stage of competition, but also gain experiences and growth that are more precious than placings~
