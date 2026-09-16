# 7th place outline

Competition: rsna-intracranial-hemorrhage-detection
Rank: #7
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117293

I make it short because there is almost no novelty in my solution.

**Overall Strategy:**
1. Train a image-level CNN and save to hard drive its GAP features.
2. Recover the original CT sequence by sorting the z-position in the meta data, and input the saved GAP features to train a scan(study)-level RNN model. 
This strategy is inspired from https://rd.springer.com/content/pdf/10.1007%2Fs00330-019-06163-2.pdf 

**Preprocessing for CNN:**
I used Appian's windowing. Spent some efforts to tweak it but results are all similar.

**Augmentation for CNN:**
Heavy augmentation including crop and resize back, affine (360 degree rotation), contrast and brightness, gamma correction, blurring and sharpening, mirroring, optical distortion, grid distortion, elastic transform ...

**CNN models:**
efficientnet_b5
efficientnet_b6
inception_resnet_v2
inception_v4
senet154
seresnext50
seresnext101
Totally 7 models, all trained on a different 80-20 training-validation split. The input resolution varied between 384x384 and 512x512 depending on the size of the model.

**RNN models:**
Two bidirectional GRU layers. Length of sequence fixed to 72. Padding and loss masking used.
