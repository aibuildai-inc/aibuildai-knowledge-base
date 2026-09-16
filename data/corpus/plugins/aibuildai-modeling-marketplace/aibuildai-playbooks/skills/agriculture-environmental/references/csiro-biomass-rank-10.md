# 10th Place Solution

Competition: csiro-biomass
Rank: #10
Source: https://www.kaggle.com/c/csiro-biomass/writeups/10th-place-solution

We would first like to express our gratitude to the competition host and the Kaggle staff for organizing this outstanding competition.
Below, we introduce the my solution

# Solution

| Item | Description |
| --- | --- |
| Model | DINOv3 ViT-L/16 distilled (Fine-tuned: regression head + last two transformer blocks) |
| Input | Images resized to 768×1536 |
| Output | 5 target variables + auxiliary-loss features (Height, NDVI, and whether Species is Ryegrass (Ryegrass_Clover)) |
| Loss function | Huber loss weighted the same way as the evaluation metric + auxiliary loss |
| Preprocessing | - Removed 8 samples suspected as label noise after inspection<br> - Standardized the target variables |
| CV | Grouped by State - Sampling-Date, then manually reviewed and merged similar groups. Split into 5 folds using a greedy method so that each target variable’s distribution is similar across folds.<br> CV and LB seemed correlated, so I tried to improve both. |
| Anti-shake measures | Data augmentation to reproduce unnatural artifacts present in the train set (red-dot artifacts, and a date stamp in the bottom-right corner) |
| Ensemble | Trained 3 models (with/without auxiliary losses) and averaged their predictions |
| Other | - EMA<br> - TTA (horizontal flip) |

## 1. Model, Input/Output, and Ensemble

I used DINOv3 ViT-L/16 distilled. Images were resized to 768×1536 and fed into the model. The model outputs consist of the five target variables plus optional auxiliary-loss features, and the final prediction is the average of three models with different output configurations.

[model]

| Pattern | Output configuration |
| --- | --- |
| 1 | 5 target variables |
| 2 | 5 target variables + Height + NDVI |
| 3 | 5 target variables + Height + NDVI + Species (binary classification: whether it is Ryegrass (Ryegrass_Clover)) |

## 2. Preprocessing

For samples with State=WA and Sampling_Date=2015-08-21, the label indicates Dry_Green_g = 0. However, upon inspecting the images, there were clearly plants other than Clover present. I therefore treated these as label errors and removed the following 8 samples from the training data:

```
- ID681680726.jpg
- ID697718693.jpg
- ID40849327.jpg
- ID230058600.jpg
- ID1403107574.jpg
- ID1337107565.jpg
- ID1139918758.jpg
- ID1761544403.jpg
```

~~This improved the score by +0.04 on CV, +0.05 on Public, and +0.01 on Private.~~




This improved the score by +0.01 on CV, +0.01 on Public, and +0.01 on Private.
(I corrected it because I had made a mistake.)


## 3. Anti-shake Measures

The training data contains unnatural artifacts such as the following, so I applied data augmentation to reproduce them:

* Red-dot artifacts, likely caused by flash
* The capture date/time appearing in the bottom-right corner of the image

This did not help on CV or Public, but improved the Private score by +0.01. When I tested them separately, the date/time augmentation did not improve the Public score, so this suggests the private set may contain some red-dot artifacts.

## 4. Tried but Not Effective

* Manifold Mixup
* Tuning all layers with LoRA
* Pseudo-labeling using data that was not used for training
* Using patch tokens (maybe I just used it incorrectly?)
* Using PlantCLEF2024 data / a pre-trained model (DINOv2)


# In Closing

If you have any questions, feel free to ask. 
Thank you for reading!
