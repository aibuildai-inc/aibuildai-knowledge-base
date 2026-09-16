# 41st place pytorch solution

Competition: recursion-cellular-image-classification
Rank: #41
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110444

**1. Data**
- 6 channels

**2. Augmentation**
- RandomScale, Rotate, HorizontalFlip, VerticalFlip, Resize, RandomBrightnessContrast, RandomGamma, Normalize

**3. Model design**

- Backbone: DenseNet201 pretrained on ImageNet
- Head: 2 linear layers with batch normalization

**4. Loss**
- Binary Cross Entropy Loss

**5. Training**

- Optimizer: Adam
- Different learning rates for different layers
- Image size: 512
- Batch size: 64
- Epochs: 75
- Finetuning for each cell type
- Mixed precision

**6. Prediciton**

- TTA: 10
- Use embeddings instead of final probability scores
- Run k-Nearest Neighbors for each cell type separately
- **Hungarian algorithm is used to match cell types with plates, wells with siRNAs**

**7. Result**

- Public LB: 0.701
- Private LB: 0.959

**8. Observations**

- I didn't manage to leverage ArcFace :(
- **Hungarian algorithm boosted score a lot**
- TTA helps too

GitHub link: https://github.com/rebryk/kaggle/tree/master/recursion-cellular
