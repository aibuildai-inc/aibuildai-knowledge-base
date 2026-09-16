# 5th place solution[NS embedding]

Competition: google-universal-image-embedding
Rank: #5
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359161

※2022/10/12 Title changed due to change in ranking.
※2022/10/18 Add solution code: https://github.com/riron1206/kaggle-Google-Universal-Image-Embedding-Competition-5th-Place-Solution
※2022/10/19 Add arXiv paper: https://arxiv.org/abs/2210.09495

Thank you to the hosts and the organizers for having such a challenging competition! 
Thinking about good image embedding was a lot of fun. And thank you to my teammates and all the participants for hard working!


# Key Points

- Dataset tuning
- CLIP visual encoder trained on LAION-2B
- head only training with ArcFace 
- strong regularization (weight_decay=0.1)
- use feature (normalized original height, original width, original aspect ratio)
- TTA
- resize antialias=True



# Dataset

- [motono0223's google landmark recognition 2021](https://www.kaggle.com/datasets/motono0223/guie-glr2021mini-tfrecords-label-10691-17690)
  - Randomly deleted data for 2000 classes
- [motono0223's products10k](https://www.kaggle.com/datasets/motono0223/guie-products10k-tfrecords-label-1000-10690)
  - All data were used
- [GPR1200](https://github.com/Visual-Computing/GPR1200)
  - Deleted 200 classes of iNaturalist data
- [food101](https://www.kaggle.com/datasets/srujanesanakarra/food101)
  - Reduced the number of data per class to 30


# Validation

- In order to see the performance of zeroshot, we prepared a test set with only the data of classes that do not exist in train and validation
  - train_valid test split: ```StratifiedGroupKFold(n_splits=20).split(X, y=< used Dataset >, groups=< class_label >))```
  - train valid split: ```StratifiedKFold(n_splits=20).split(X, y=< class_label >))```
- In the submit, we used the weights of the epochs with higher MAP5 in the test set


# Model

It is very important to use ViT trained with LAION-2B from [OpenCLIP](https://github.com/mlfoundations/open_clip) for the backbone of ArcFace. OpenCLIP is more powerful than the official [openai/CLIP]( https://github.com/openai/CLIP).

Also, adding image size information to embedding improved public lb by about 0.003.

- **backbone: OpenCLIP ViT-H/14 224x224**
- head: BatchNorm1d(num_features=1024+3, affine=False) -> Linear(num_features=1024+3, out_features=64)
  - num_features is set to +3 because normalized original height, original width, original aspect ratio are concatenated in the embedding of the backbone
- loss: ArcFace(scale=30.0, margin=0.5)

[model_architecture]

# Training

It is important to apply strong regularization(weight decay) to the model parameters to avoid overfitting the training data.

- **freeze backbone parameters and train only head**

- optimizer: SAM(AdamW(**weight_decay=0.1**))

- scheduler: warmup 0-3epoch(lr=0.0001->0.01) + Cosine Annealing 3-1000epoch(lr=0.01->0.0001)

- augmentation: we used Albumentations follow OpenCLIP augmentations

  ```
  A.Compose([
      A.RandomResizedCrop(height=224, width=224, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
      A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], max_pixel_value=255.0, p=1.0),
      ToTensorV2(),
  ])
  ```


# TTA

Since lb improved when images with different heights and widths were resized keeping the aspect ratio, TTA was switched according to the aspect ratio of the image as follows.

In addition, we switched TTA with a classification model. The classification model was trained with [130k-images-512x512-universal-image-embeddings](https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings).

```
import torch
import torch.nn as nn
from torchvision import transforms

class CLIPTTA(nn.Module):
    def __init__(self, encoder, neck, classifier):
        super().__init__()
        self.encoder = encoder  # torchscript OpenClip ViT
        self.neck = neck  # Model neck(head)
        self.classifier = classifier  # Classifier Model

    ...
    
    def concat_image_info(self, emb, H: int, W: int):
        subfeature = torch.tensor([[
            float(H/W),
            float(H/224.0), 
            float(W/224.0),
        ]], device=emb.device).float()
        return torch.cat([emb, subfeature], dim=1)

    def forward(self, x):
        B,C,H,W = x.shape
        ratio = W/H

        if ratio==1.0:
            x_resize = self.ccrop_resize(x,cratio=0.9)  # center_crop_resize
            feat = self.encoder(x_resize)  # OpenClip ViT embedding
            feat = self.concat_image_info(feat, H, W)
            feat = self.head(feat)
            return torch.nn.functional.normalize(feat)

        else:
            x_resize = self.ccrop_resize(x,cratio=1.0)  # resize
            feat1 = self.encoder(x_resize)  # OpenClip ViT embedding
            cls = torch.argmax(self.classifier(feat1))  # Classification
            feat1 = self.concat_image_info(feat1, H, W)
            feat1 = self.head(feat1)
            
            # cls: apparel=0, packaged=8, toys=10
            if cls == 0 or cls == 8 or cls == 10:
                x_pad = self.pad_ccrop_resize(x,cratio=0.9)  # keep aspect center_crop_resize
                feat2 = self.encoder(x_pad)  # OpenClip ViT embedding
                feat2 = self.concat_image_info(feat2, H, W)
                feat2 = self.head(feat2)

                x_pad = self.pad_ccrop_resize(x,cratio=1.0)  # keep aspect resize
                feat3 = self.encoder(x_pad)  # OpenClip ViT embedding
                feat3 = self.concat_image_info(feat3, H, W)
                feat3 = self.head(feat3)
                return torch.nn.functional.normalize((feat1+feat2+feat3)/3.0)
            else:
                x_pad = self.pad_ccrop_resize(x,cratio=0.9)  # keep aspect center_crop_resize
                feat2 = self.encoder(x_pad)  # OpenClip ViT embedding
                feat2 = self.concat_image_info(feat2, H, W)
                feat2 = self.head(feat2)

                return torch.nn.functional.normalize((feat1+feat2)/2.0)
```

Also, setting ```antialias=True``` in torchvision's resize improved lb. This is because antialias is active when the openclip model is trained.

```
torchvision.transforms.functional.resize(x, [224, 224],interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
```

# LB update history

The yellow line is the border of the GOLD MEDAL.

[lb_update]

# What did not work

- Other Dataset ([imagenet1k](https://www.kaggle.com/datasets/motono0223/guie-imagenet1k-mini1-tfrecords-label-0-999), [MET](https://www.kaggle.com/datasets/dschettler8845/the-met-dataset), [OmniBenchmark](https://github.com/ZhangYuanhan-AI/OmniBenchmark) etc...)
- fine tuning([LP-FT](https://arxiv.org/abs/2202.10054))
- ArcFace + PCA (compress output of model to 64 dimensions)
- Mixup
- sub-center ArcFace, ElasticFace, AdaFace, DistanceMarginLayer
- multi-layered head
- Concat output of timm model to embedding
- Concat output of ViT projection layer to embedding
- Model ensemble (average embedding of 64 dimensions)
- and more.....
