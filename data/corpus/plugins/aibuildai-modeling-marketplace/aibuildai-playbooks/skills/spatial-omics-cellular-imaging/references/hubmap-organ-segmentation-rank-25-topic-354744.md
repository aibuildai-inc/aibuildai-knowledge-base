# 25th Simple Solution

Competition: hubmap-organ-segmentation
Rank: #25
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/354744

I would like to thank the organizers for hosting the great competition.
also, I would like to express my gratitude to our teammate @hwigeon and new teammate @methyl for their dedication to the competition.
I also want to express my greatest gratitude to the frog @hengck23. Without his comments through the open discusstion, I would have given up this competition.    
  
# Augmentations
I can't cross even 0.6 without insane augmentations.  
After I applied very strong augmentations with a lot of epochs, I could stably score over 0.75+ 

I applied HSV, BRIGHTNESS, RANDOMCROPANDRESIZE, STAIN NORMALIZATION, AUGRESSIVE SIZE CHANGE OF PROSTATE with 500 epoch patch training + 400 epoch whole region finetuning
after these augmentations, we can score over 0.75 with most of our models

# Models  
Kaggler : ( Patch + Finetune ) Efficientnet7+Deeplabv3plus, Efficientnet6+Deeplabv3plus, Efficientnetv2l+Deeplabv3Plus, Coat-small + Coat-small-pl [768,1536, 1024,1536,1536]
Hwigeon : (Only Patch) swin transformer_Unet   + PvT + Coat
Methyl : (Only whole image) PVT + Coat  [1024,1024,1024,1024]
  

# Validation  
We failed validation. When teaming up, though I know validation is very important, but with time being constrained or as the performances of some models are not totally reproduced, we can't validate our cv properly. We just depended on hubmap LB. 

# Threshold
organ_threshold = {
        'Hubmap': {
            'kidney'        : 0.40,
            'prostate'      : 0.40,
            'largeintestine': 0.40,
            'spleen'        : 0.40,
            'lung'          : 0.10,
        }}
  
We failed to get the gold medal but There is no regret since I tried my best  
Thank you for reading this!
