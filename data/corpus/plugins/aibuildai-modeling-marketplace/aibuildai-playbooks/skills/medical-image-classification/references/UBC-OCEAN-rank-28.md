# 28th solution

Competition: UBC-OCEAN
Rank: #28
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465425

It was an interesting competition and I have learned many techniques and insights from others!
Thanks to the competition host, Kaggle, and @jirkaborovec for nice train and inference code.

# Overview of my approach
- TMA images : center crop and tiling and inference with TMA model 
- WSI images : tile the WSI thumbnail images and classify tumor area with tumor classifier model, and extract tiles from original WSI images and classify sub-types with TMA model 
- [train/inference code](https://github.com/jinhopark8345/UBC-OCEAN-30th-place-solution) 

# TMA pipeline
- Train 
    - TMA model : extract tiles from WSI images with supplemental masks (crop size : 1024x1024 -> resize 512x512) and fine-tune [maxvit_tiny_tf_512.in1k](https://huggingface.co/timm/maxvit_tiny_tf_512.in1k)
(For TMA model training, I used tiles with more than 70% cancerous tumor pixels and for validation, 30% ~ 70% tumor pixels)
- Inference 
    - cropped and resized TMA tiles (extract tiles from TMA images with 2048x2048 resolution, and resize them to 512x512, stride 256, zoom : x40->x10)
    - inference with TMA model -> each tile with predicted ovarian sub type
    - majority votes and make final prediction

# WSI pipeline detail
- Train
    - Tumor classifier : TMA model but with WSI thumbnails and compressed WSI supplemental masks
    - (TMA model : used the same TMA model from TMA pipeline)

- Inference
    - tile WSI thumbnail image
    - inference with Tumor classifier -> each thumbnail tile with tumor or non-tumor result
        - no tumor tiles -> 'Other'
        - tumor tiles -> center crop and pass it to TMA model and do majority votes and make final prediction

# Tried but didn't work
- [StainNet](https://github.com/khtao/StainNet)  did not get better result than simple normalization
