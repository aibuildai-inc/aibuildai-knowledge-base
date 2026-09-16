# 38th solution (Private 0.52, Hight score 0.55)

Competition: UBC-OCEAN
Rank: #38
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465368

Since I first started learning about data analytics, I've heard about Kaggle from many people and have come to admire them. If this competition ends successfully, I will become a competition master two years after starting Kaggle! Thanks everyone!

And 'Gunes Evitan''s pyvips code was very helpful during the competition. Thank you.
[https://www.kaggle.com/code/gunesevitan/libvips-pyvips-installation-and-getting-started](url)

### Preprocessing
After using the back ground provided by the competition, we applied the otsu threshold. And I cut the image to size 512 x 512 and saved it.

### Training
- Model : VIT-s + TransMIL
- Augmentation : VerticalFlip, HorizontalFlip, CLAHE, RandomGamma, GridDistortion, ShiftScaleRotate
- Optimizer & learning rate: Since vis-s was already pre-trained and MIL was prone to overfitting, vit-s was trained with a learning rate of 1e-6 and MIL was trained at a learning rate of 1e-5, and EMA was applied to each. AdamW and CE were used.

>optimizer = torch.optim.AdamW([{'params': model.image_extractor.parameters(),'lr':1e-6}, {'params': model.mil.parameters()}], lr=1e-5, weight_decay=1e-3)
extractor_ema = ModelEma(model.image_extractor, decay=ema_decay, device=None, resume='')
mil_ema = ModelEma(model.mil, decay=ema_decay, device=None, resume='')

I experimented with two methods.

1. Traning only MIL: A weakly supervised method that extracts features from patch images using the vit-s model and then learns using only those features.
2. Training with image encoder (vit-s) together: We randomly selected 100 images from a 512x512 patch for learning and evaluation.

Of the two, method 2 showed better pb score.

### Tried(helpful)
- Pseudo-labeling 1536x1536 : After pseudo labeling the 1536x1536 image using MIL learned at 512x512, we learned a model for TMA prediction using images with a probability of 0.5 or higher. Although it was not good in pb score, it achieved 0.55 in private.
- Outlier detect: Each class was learned using binary cross entropy. After applying sigmoid, if all class predictions were less than 0.5, it was predicted as 'Other'. It's not exact, but there was an increase of about 0.1.
- Upscaling : It was better than applying weights to cross entropy.

### Tried(but didn't help)
- staintools: augmentation with staintools. But it didn't help much.
-  Other dataset(external) : [https://www.cancerimagingarchive.net/collection/ovarian-bevacizumab-response/](url) In this dataset, I trained a model with the class corresponding to 'UC' as other, but it did not help at all.
