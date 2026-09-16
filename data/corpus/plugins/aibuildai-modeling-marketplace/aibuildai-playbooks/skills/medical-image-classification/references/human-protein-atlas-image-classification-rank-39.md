# 39th solution-Attention Gated Resnet18 ( single model without cv)

Competition: human-protein-atlas-image-classification
Rank: #39
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77637

Our solution is based on the Attention Gated Network (AGN). In our model, Resnet18 is used as the backbone and feature maps of the last 3 blocks are used to generate attention gate. Given the attended features at last 3 blocks of Resnet18, we combine them for final prediction by average mean.We random crop the original image(512x512) into 3 different size (256,384,512) to fit 3 different AGN model, and finally ensemble their predict prob on full size (512x512) with threshold 0.2 as the final results. Our solution used only single model and 512x512 PNG files with HPA external  data, and did't use cross validation.

Attention Gated Network (AGN): https://arxiv.org/pdf/1804.05338.pdf
![Attention Gated Network][1]
![Attention Unit][2]

**Dataset**
kaggle data and HPA external data(512x512 RGBY), split by Multilabel Stratification Python Package, not use TIFF images 

**Training methods**
Simply training, SGD with momentum, learning rate = 0.1, ReduceLROnPlateau lr scheduler.

**Loss functions**
The sum of soft f1 loss and focal loss

**Data augmentation**
Random flip and random crop.

**TTA**
Random flip

**Result**
ensemble three image size: 0.604| 0.540
ensemble three image size and oversample(size 256): 0.601|0.547


  [1]: https://drive.google.com/file/d/18lVqM3YEI2Z6u-b3LFT5zqr6gKR8_Dm_/view
  [2]: https://drive.google.com/file/d/1zSJ1KZOIn-ngS3LPGL-kOROL7VOcUAG3/view
