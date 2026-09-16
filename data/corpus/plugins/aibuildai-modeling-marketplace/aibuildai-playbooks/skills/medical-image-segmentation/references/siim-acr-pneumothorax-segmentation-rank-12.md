# 12th place solution: UNet with mask scoring head

Competition: siim-acr-pneumothorax-segmentation
Rank: #12
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107757

Congratulations everyone!
And thank you for my teammates @masaakikob and @card7077 !

I found something interesting and would like to share it.  
The final submission was made only from my models.

# 1. My strategy

- Predict Segmentaion
  - img size 896
- Predict Class
  - img size 768
- Integrate Predicts
  - Remove low probability masks


# 2. Model &amp; Dataset Overview

## 2.1 Data flow

MS-EUNet means "Mask Scoring UNet with EfifcientNet B4" in this term.  
TTA is horizontal flip.





## 2.2 About Dataset

| Dataset | Description |
|:--|:--|
| Stage1 Train | Kaggle stage 1 train set |
| Stage1 Test | Kaggle stage 1 test set |
| Stage2 Train | Kaggle stage 2 train set(= Stage 1 Train + Stage1 Test) |
| CheXpert | External Data. Removed ambigous labels. All image: 210k, pneumothorax images: 17k |
| CheXpert_1 | Sampled from CheXpert, 50k, pneumothorax : not pneumothorax = 1 : 2 |
| CheXpert_2 | Sampled from CheXpert, 68k, pneumothorax : not pneumothorax = 1 : 3 |
| DatasetSeg_1 | Only pneumothorax data in "Stage1 Train", 3k |
| DatasetSeg_2 | Only pneumothorax data in "Stage2 Train", 3k |
| DatasetCls_1 | Stage1 Train + CheXpert_1, 67k |
| DatasetCls_2 | Stage1 Train + CheXpert_2, 67k |
| DatasetCls_all | Stage1 Train + CheXpert, 220k |

## 2.3 Which model was trained on which dataset

| Model | Used Dataset |
|:--|:--|
| SEResNext(Model1) | DatasetCls_1 |
| SEResNext(Model2) | DatasetCls_2 |
| LightGBM | DatasetCls_all |
| MS-EUNet | DatasetSeg_2 |
| MS-EUNet(with ASPP) | DatasetSeg_1 |

# 3. Segmentation

## 3.1 Mask Scoring Head(little different from original)

- This is based on Mask Scoring R-CNN (MS R-CNN)
  - http://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf
- I was interested in applying this idea to UNet architecture
- I repeated the experiment. As a result, this was a useful case for my model



```python
class UEfficientNetB4WithMaskScoreHead(nn.Module):
    def __init__(self, dropout_rate=0.3, input_size=256):
        super(UEfficientNetB4WithMaskScoreHead, self).__init__()
        #############
        # ~~~~~~~~~ #
        #############
        kwargs = {'mask_ch': 1, 'num_classes': 1, 'mask_feature_size': 32}  # 32 is better than 56
        self.mask_iou_head = ROIMaskScoreHead(in_feature_ch=in_feature_ch, **kwargs)

    def forward(self, x):
        #############
        # ~~~~~~~~~ #
        #############
        deconv1 = self.decoder2(upconv2)
        upconv1 = torch.cat([deconv1, conv1], 1)  # size: input_size//2
        upconv1 = self.dropout(upconv1)

        deconv0 = self.decoder0(upconv1)
        upconv0 = self.dropout0(deconv0)

        output_mask = self.output_layer(upconv0)

        # Mask score calc
        output_score = self.mask_iou_head(upconv1, output_mask)
        return output_mask, output_score
```


## 3.2 Loss function for Mask Scoring UNet


```python
class BCEDiceLoss(nn.Module):
    """
    Loss defined as alpha * BCELoss - (1 - alpha) * DiceLoss
    """
    def __init__(self, alpha=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        loss = self.alpha * bce_loss + (1. - self.alpha) * dice_loss
        return loss


class DiceScoreL2Loss(nn.Module):
    """
    Loss for mask scoring UNet

    Caution: This loss for 1 class !!!!!
    """
    def __init__(self):
        super(DiceScoreL2Loss, self).__init__()
        self.smooth = 1e-8
        self.sum_dim = (2, 3)

    def forward(self, dice_logits, logits, masks_gt):
        dice_preds = torch.sigmoid(dice_logits)

        # BATCH x 1 x H x W
        masks_preds = torch.sigmoid(logits)

        # BATCH x 1
        intersect = (masks_preds * masks_gt).sum(self.sum_dim).float()
        union = (masks_preds.sum(self.sum_dim) + masks_gt.sum(self.sum_dim))

        # Calc dice(BATCH x 1)
        dice_gt = (2. * intersect + self.smooth) / (union + self.smooth)

        # Calc L2 Loss
        cond = torch.abs(dice_gt - dice_preds)
        loss = 0.5 * cond ** 2
        return loss.mean()

loss_func1 = BCEDiceLoss(alpha=0.7)
loss_func2 = DiceScoreL2Loss()
loss = loss_func1(logits, masks) + 0.2 * loss_func2(logits_score, logits, masks)
```

## 3.3 Training params





## 3.4 Ablation Study

- Dataset: Stage2 Train set(Stage1 Train + Stage1 Test)
- Only pneumothorax data.




# 4. Classification

## 4.1 Train 2 binary classifier(seresnext101) by DatasetCls_1 and DatasetCls_2




```python
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
milestones = [int(args.num_epochs * r) for r in [0.6, 0.7, 0.8, 0.9]]
scheduler = lr_scheduler.MultiStepLR(
    optimizer, milestones, gamma=0.5, last_epoch=-1)
```


## 4.2 Get Embeddings with TTA

Concat all embeddings, I can got 8,192 dimensional features.

## 4.3 Add dicom info

- Sex: 0 or 1
- Age: No change
- ViewPoint: 0 or 1

I can make 8,195 dimensional features for each images in DatasetCls_all.

## 4.4 Train binary classifier(LightGBM) with optuna


# Last

P.S. We're hiring!: https://lpixel.net/en/careers/
