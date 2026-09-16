# 10th Place Solution

Competition: google-universal-image-embedding
Rank: #10
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359271

Congrats to all the winners, and thanks to organizers and all participants. This is a joint writeup with my wonderful teammate @yamash73 and @ksork6s4 (thanks to awesome team work).

**Things that worked**
- Selection of dataset
- BatchNorm1d in head
- Ensemble of OpenCLIP backbones

**Dataset**
We have noticed that Public LB scores vary greatly depending on the combination of datasets. Therefore, we are exploring the best combination of data sets in the early stages of the competition. Comparison of experimental results shows that "Inshop", "Instre" and "Consumer-to-shop" are effective. Based on these results, we have decided to use "Products10k", "Landmark", "Instre" and "Consumer-to-shop" as our basic dataset combination . Here "Inshop" and "Consumer-to-shop" are subsets of deepfashion, so only "Consumer-to-shop" is selected as our combination of datasets.



NOTE:
- The results of separate experiments confirm that Products10k and Landmark contribute to the LB scores
- For Inshop, Stanford products, and Consumer2shop, classes with a sample size more than 5 are used

Our final combination of datasets is as follows:
- Products10k 
- Landmark
- Instre
- Consumer-to-shop
- Food101 (added in the last week)

**Model**
The backbone of our final model is an ensemble of 3 OpenCLIP image encoders. OpenCLIP backbones contributed to the huge improvement of the LB score. We refered to the head/loss design from the past competitions (mainly Google Landmark Retrieval 2021).
- Backbone: OpenCLIP ViT-B + OpenCLIP ViT-L + OpenCLIP ViT-H
- Head: Linear + BatchNorm1d + PReLU + subcenter ArcFace(k=3)
- Loss: ArcFace Loss with dynamic margin (s=30.0, mx=0.65, my=0.15) + focal loss



**Training**
Only head is trained like other teams. At first, we set the total epochs to 20, but after experiments showed that we could achieve the same level of performance with 5 epochs, we were able to increase the number of trials.
- Total epochs: 5
- Batch size: 512
- Optimizer: Adam (lr=1e-2, weight decay=1e-4)
- Scheduler: CosineAnnealingLR (eta_min=1e-3, T_max=5)
- Augmentation: RandomResizedCrop, HorizontalFlip, ImageCompression, HueSaturationValue, RandomBrightnessContrast, Blur, ShiftScaleRotate, CoarseDropout
	
**Things that didn't work**
- Dimensional reduction of embeddings using PCA
- Training OpenCLIP ViT without freezing backbones
- Cropping with saliency models (preprocess)
- TTA (horizontal flip)
- AutoEncoder
- Other architectures of head
