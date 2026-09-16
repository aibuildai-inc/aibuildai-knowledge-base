# 3rd Place Solution

Competition: sorghum-id-fgvc-9
Rank: #3
Source: https://www.kaggle.com/c/sorghum-id-fgvc-9/discussion/328593

Congrats to all the winners.
Thanks to Kaggle and the hosting team for an interesting competition. 
I appreciate my team member @SisuoLYU @haogood666 !

#  **Dataset**
Through data set analysis, it can be found that the data of this question has the following classification difficulties.
1 There is a high degree of inter-class visual similarity between different categories, and fine-grained classification is difficult.
2 Outdoor pictures on different dates and at different times of the day have large differences in lighting, and there are many exposed images.
3 The training set and the test set are from sorghum grown on two different fields, and there is a problem of domain adaptation.
4 With the growth time, the state and height of plants are changing.
Finally, our training size is:1024x1024. Dataset We convert png format to jpeg format, which improves storage space. Thanks to  @
Mithil Salunkhe 
 https://www.kaggle.com/competitions/sorghum-id-fgvc-9/discussion/313266 for the open source jpeg data.
It is mentioned here that we perform image histogram equalization preprocessing for both training and testing, which improves the problem of exposed images. Here, we are very grateful to@Jun-Ming Chen 
https://www.kaggle.com/code/leoooo333/lb-0-885-sorghum-higer-accuracy open source notebook. We referenced many of their ideas and proposals.

# **Model**



Baseline: The initial debugging is trained on 512x512 images, resnet50 model, CrossEntropyLoss, cosine restart learning rate plan, AdamW optimizer.
Summarize:

| methods |scores |
| --- | --- |
|  base(resnet50,512x512)| 0.73 |
|  histogram equalization| +0.03 |
|  Ibn-resnet50| +0.05 |
| bnn-neck| 	+0.005| 
| The last CNN layer stride is changed to 1| 	+0.005| 
| Arcface(s=30,m=0.3)	| +0.015| 
| mixup	| +0.01| 
| cutmix	| +0.005| 
| Awp adversarial training| 	+0.01| 
| 512x512->1024x1024| 	+0.04| 
| fgvc8data| 	+0.03| 
| tta	| +0.02| 
| pse_udo| 	+0.01| 
| total	| ≈0.957| 

Ibn-resnet：IBN-Net integrates BN and IN, which can improve the learning ability and generalization ability of the model. The design principle is: use both IN and BN in the shallow layers of the network, and only use BN in the deep layers of the network. In this problem, since the training set and the test set come from two different geographical locations, the addition of IN greatly improves the model performance. 


Arcface：The topic of this competition is fine-grained visual image classification, with high similarity between classes. Ordinary Softmax Loss does not explicitly optimize feature Embedding to enhance the similarity of intra-class samples and the inconsistency of inter-class samples. ArcFace is an additive angular margin penalty that adds an additive angular margin m between the feature and target weights to enhance both intra-class compactness and inter-class dissimilarity.

Awp：Adversarial training is used in nlp to improve the performance of the model, such as fgm, pgd, but because it is the disturbance added in the text emb, in the image field, we choose the general awp confrontation training, which can add disturbance to the model weight and input at the same time, This increases the robustness of the model.

# **Others**
Data augmentation mainly combines the characteristics of this data illumination change. In addition to mixup and cutmix, the following data augmentation:
```
train_transform = A.Compose([
         A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
         A.Resize(CFG.image_size, CFG.image_size),
         A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.5),
         A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
         A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)]),
         A.Normalize(),
         ToTensorV2(p=1.0),
     ])

val_test_transform = A.Compose([
         A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
         A.Resize(CFG.image_size, CFG.image_size),
         A.Normalize(),
         ToTensorV2(p=1.0),
     ])
```
Another thing worth noting in this question is that we used the fgvc9 and part of the fgvc8 training dataset. Since the image size of the fgvc8 dataset is 480x480, we spelled 1024x1024. 

# **Some failed attempts**

1 The model is changed to efficient, vit, etc.
2 The model fusion has not been scored, it may be that the late score is already very high, and the model fusion has not been improved.
3 The various improvements of loss, such as focalloss and softCrossEntropy, have not changed much.

# **Code Links**

https://github.com/xinchenzju/CV-competition-arsenal/tree/main/CVPR2022-FGVC9-top3
