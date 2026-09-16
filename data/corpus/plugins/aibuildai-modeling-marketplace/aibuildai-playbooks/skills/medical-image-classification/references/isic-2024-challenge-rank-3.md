# 3rd Place Solution

Competition: isic-2024-challenge
Rank: #3
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532919

I’m thrilled to have placed 3rd in this competition and would like to thank Kaggle and the organizers for making this amazing experience possible. Congratulations to all the other winners as well!

Special thanks to [@greysky](https://www.kaggle.com/greysky) [@murashow](https://www.kaggle.com/murashow) [@merfarukelik](https://www.kaggle.com/merfarukelik) [@richolson](https://www.kaggle.com/richolson) for their work and sharing throughout the competition.

# Solution Overview
My solution, in line with several other outstanding public notebooks, integrates GBDT tabular models with outputs of image models as features. Most of my work focused on developing a variety of image models to boost the GBDT score. As many kagglers have pointed out, the key factor in this competition is trust-CV, I focused on creating reliable CV and maximizing the CV score. (My final CV is 0.183099)


# Image model
My final solution uses 4 image models

**Positive: "target" = 1**
- convnextv2_nano.fcmae_ft_in22k_in1k (CV 0.1599389)
- vit_tiny_patch16_224.augreg_in21k_ft_in1k (CV 0.1612504)

**Positive: target=1 + iddx_1=Indeterminate + iddx_2!=nan**
- vit_tiny_patch16_224.augreg_in21k_ft_in1k (CV 0.1460650)
- vit_small_patch16_224.augreg_in21k_ft_in1k (CV 0.1486832)

The last 2 models above are aimed to give conservateive prediction so that GBDT models can have more variety of features.
Overall, as many have mentioned, smaller models performed better in my case as well.

## Training technique
- scheduler: CosineAnnealing with warmup
- batch_size: 32
- learning rate: 1e-4
- optimizer: AdamW
- weight decay: 0.001 for weight param
- positive: negative = 1:1
- Classification head: fc(64 or 32) + relu + dropout
- Change negative samples per epoch
- Augumentation based on 1st solution in prev comp [1st place solution for SIIM-ISIC Melanoma Classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412)
- Ensure the same pos/neg ratio in every batch (sample code below)

  ```python
  class ISICDataset(Dataset):
      def __init__(self, hdf5_file, isic_ids, targets=None, transform=None, ratio_int=2):
          self.hdf5_file = hdf5_file
          self.isic_ids = isic_ids
          self.targets = targets
          self.transform = transform
          self.ratio_int = ratio_int  # If ratio_int=2 then pos:neg = 1:2
          self.positive_list = [ii for ii, tt in zip(self.isic_ids, self.targets) if tt == 1]
          random.shuffle(self.positive_list)
          self.negative_list = [ii for ii, tt in zip(self.isic_ids, self.targets) if tt == 0]
          random.shuffle(self.negative_list)
          self.balanced_list = self.create_balanced_list()

      def create_balanced_list(self):
          balanced_list = []
          pos_count = 0
          neg_count = 0
          # Repeat and arrange the Positive list and Negative list in sequence according to a specified ratio.
          while pos_count < len(self.positive_list) or neg_count < len(self.negative_list):
              if pos_count < len(self.positive_list):
                  balanced_list.append(self.positive_list[pos_count])
                  pos_count += 1

              for _ in range(self.ratio_int):
                  if neg_count < len(self.negative_list):
                      balanced_list.append(self.negative_list[neg_count])
                      neg_count += 1
          return balanced_list

      def __getitem__(self, idx):
          isic_id = self.balanced_list[idx]
  ```



# Cross Validation strategy
GroupStratified 5Fold by patient_id and checked the patient number and positive labels are evenly distributed across the folds. Same fold split is used for both image and tabular model.

# Tabular model
My tabular model is almost identical to the following great notebooks. And my final solution uses LGBM(w/o image) + LGBM(w image) + Catboost(w image) + XGB(w image).
https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data
https://www.kaggle.com/code/murashow/tabular-with-image-features-lightgbm
https://www.kaggle.com/code/merfarukelik/tabular-with-image-features

# Things didn't work
### Image model which predict the difference between target and GBDT prediction
Throughout the competition, I kept thinking about how to best integrate the GBDT model with the image model, aiming for them to complement each other’s weaknesses by covering the areas where the other struggled. I tried training the image model with the target as ("target" - "GBDT prediction"), but this model did not contribute to the final integrated GBDT model.
### Feature engineering - merge left and right
In my opinion, it didn’t seem logical that the probability of a malignant occurrence would differ between the left and right sides of the body. So, I created new features by combining categories like "Left Arm - Lower" and "Right Arm - Lower" into a single "Arm - Lower" feature, but this did not lead to any score improvement.

### Others
- Layerwise learning rate decay destroyed training.
- Models like Eva02, Swin, and EfficientNet were not significantly better.
- Stacking of GBDT models
- More image models(>4)


Lastly, my work is built upon the contributions of many Kagglers. Once again, I want to express my gratitude to all the Kagglers who have generously shared their knowledge and techniques. It’s been a great learning experience, and I’m excited to continue growing with this fantastic community.

# Code
My code is available [here](https://github.com/kyohei-123/kaggle-isic-2024-3rd-place-solution)
