# 48th place solution

Competition: hubmap-organ-segmentation
Rank: #48
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/355509

Thanks to  the organizers for hosting such a great competition. 
I want to express my greatest gratitude to the @hengck23 Without his comments through the discusstions, I learned a lot in the competition from his opinion!

# Augmentations
- Because i applied different ensemble model to different model, so there was a few different augmentations.

## Used augmentation:
- 5 simple augmentation: HSV, RANDOM FLIPS, Rotate 90 DEGREES, RANDOM NOISES, RANDOM CONTRAST
- Stain augmentation for specific organs (spleen, lung), i also use this augmentation to prostate, but it never got better score.

## Different augmentation for different organs
- 5 simple augmentation for 3 organs (prostate, largeintestine, kidney)
- 5 simple augmentation and Stain augmentation for 2 organs (spleen, lung) 

# Models (different ensemble models for each organ)
-  Training image size: 768 * 768
### kidney, largeintestine
- pvtv2-b4 backbone with conv3x3 decoder, 3 of 5folds (fold1,fold2,fold3),each weight are 0.05, 0.3, 0.65.
### prostate (using ensemble models didn't get better result to public score)
- pvtv2-b4 backbone with conv3x3 decoder, 1 of 5folds (fold3)
### spleen
- pvtv2-b4 backbone with conv3x3 decode, 3 of 5folds (fold1,fold2,fold3) + 1 of 4 folds for spleen only (fold3), and just averaged all model predictions
### lung 
- pvtv2-b4 backbone with conv3x3 decode, 1 of 5folds (fold2, fold3) + 
- pvtv2-b4 backbone with DAformer decode, 2 of 5folds lung only(fold2, fold3) +
- pvtv2-b4 backbone with conv3x3 decode, 1 of 5folds lung only, all training images are stain normalized(fold3)
- weights: 0.28 in pvtv2-b4 backbone with conv3x3 decode of fold3, others are 0.18

# Validation
- Using 25% training data to validation for one organ only
- Using 20% training data to validation for all organ training

# Threshold

organ_threshold = {

    'Hubmap': {
        'kidney'        : 0.45, 
        'prostate'      : 0.40,                    
        'largeintestine': 0.30,                  
        'spleen'        : 0.30,                      
        'lung'          : 0.07,                         
    },

    'HPA': {
        'kidney'        : 0.50,
        'prostate'      : 0.50,
        'largeintestine': 0.50,
        'spleen'        : 0.50,
        'lung'          : 0.10,
    },
}

# Detailed notebook link
https://www.kaggle.com/code/chris666/48th-place-inference
