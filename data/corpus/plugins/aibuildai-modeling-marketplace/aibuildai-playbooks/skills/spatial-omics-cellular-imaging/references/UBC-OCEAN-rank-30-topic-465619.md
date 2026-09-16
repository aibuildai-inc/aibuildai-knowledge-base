# 30th place solution

Competition: UBC-OCEAN
Rank: #30
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465619

This was my first Kaggle competition and it was a fun ride. Many thanks for the organizers and everyone involved. Special thanks to @jirkaborovec whose public notebooks helped me a lot at the beginning of this competition to kickstart my approach. 

# Best model
I used a simple tile-based approach where I trained a ConvNeXt_Tiny model to classify each tile as one of the 5 categories:
- ConvNeXt_Tiny, pretrained
- 512x512 px image size
- batch size: 32
- random horizontal/vertical flip augmentation
- color augmentation (brightness, contrast, saturation, hue)
- color normalization
- Cross-Entropy Loss
- AdamW optimizer, StepLR (step_size=2, gamma=0.1), LR=1e-4
- 1 + 5 epochs (convolutional layers feezed for the initial epoch)
- CV5, Stratified Group KFold

# WSI tiling
Using the supplemental masks, I trained an EfficientNet_B0 (pretrained) model with similar training parameters as the main model to detect tiles with tumor (0.97 CV5 balanced accuracy). As training data, I selected tiles with tumor label > 95% for the tumor class and tiles with stroma+necrosis label > 50% and tumor label < 5% for the no-tumor class.

I cut each WSI into 1024x1024 px tiles (dropping tiles with > 60 % black background) and used 32 random tumor tiles (based on the EfficientNet_B0 model with 0.5 threshold) from each image for both training and inference. Predictions were averaged between tiles during inference.

# TMA “tiling”
One 2048x2048 px (to compensate for the x2 magnification) tile was cut from the center of the image.

# Other class
Using a sigmoid activation function, an image was marked as Other if the largest activation was smaller than 0.8.



(As an extra: some accidental augmented images with questionable artistic value.)
