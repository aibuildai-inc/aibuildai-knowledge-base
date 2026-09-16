# The 3rd place solution

Competition: siim-acr-pneumothorax-segmentation
Rank: #3
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107981

Congrats to all the winners, thanks Imaging Informatics in Medicine (SIIM),American College of Radiology (ACR), @RadiologyACR, Society of Thoracic Radiology (STR), @thoracicrad and Kaggle 
 for  hosting such a competition which may aid in the early recognition of pneumothoraces and save lives.


**Challenges:**
1. Relative large image size.
2. The quality of the external data sets are not so good, but make full use of them may help.


**Solution:**

[pipeline]

**Data:**
As the image sizes are little big, to save memory and computation, I first trained an UNET model to predict lungs from 1024x1024 images, all my models were based on the cropped lungs,  576x576 cropped images were good enough for my models.
[lung segmentation unet]

In deep learning tasks, more data are often important, so I tried to make full use of CheXpert and NIH datasets,  we could not use them directly after I  had read the papers related as the labels of them are not accurate enough, I guessed that I could use pseudo labels, so I used a resnet34 UNET model  which trained on competition data (0.858 on public LB) to predict CheXpert positive samples, and selected the positive samples which also predicted by my model as positive samples when training pseudo label models, as time was limited after the postponed  DOGs-GAN competition, I did not predict the negative samples of the CheXpert, just treated them as negative.  As to NIH dataset, I did not use the labels they provided, just believe my models predictions. 
[data strategy]
I kept ratio of positive and negative samples equal, when training pseudo label models, I kept the ratio of the pseudo samples be 0.5 of normal samples.

**Models**

Experiments  were done mainly on UNET based on resnet34 and SE-resnext50 backbones, I selected them because resnet34 is light enough to do experiments and SE-resnext50 is deep enough for this competition, I did not have enough time and resources to train larger and deeper models. 
[seresnext]
My final models were 3 SE-Resnext50 models:  
m1:704x704 images with no pseudo 
m2:576x576 images with CheXpert Pseudo
m3:576x576 images with  CheXpert and NIH Pseudo 

Although my final submission  were ensemble of these three models, but 3 of 10 folds of m2 could reached 0.8809 on public LB and 0.8642 on private LB. The ensemble did not bring too much improvements in this competition.

Attention: CBAM
Loss:  Lovasz Loss, I tried Active Contour Loss also, but did not bring local CV increase.
No classification model,No classification loss, I thought  pixel level labels were enough and if we introduced classification model, it's hard to select thresholds on the private dataset.
No threshold search, just used 0.5
Optimizer: Adam with 0.0001 learning rate, no learning rate change during training
Epochs: 15
EMA of model parameters after 6 epoch.
Batch Size: 3 per GPU when trained on 576x576 images, so the batch size was 9 when using 3 1080Ti, no accumulations of batch-size, no Synch-BN was used. 2 per GPU on 704x704 images.


https://github.com/bestfitting/kaggle/tree/master/siim_acr
