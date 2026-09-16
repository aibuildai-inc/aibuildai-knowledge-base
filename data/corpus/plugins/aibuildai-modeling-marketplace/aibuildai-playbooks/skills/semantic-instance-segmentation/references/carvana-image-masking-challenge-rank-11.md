# LB 11th(0.9971) solution overview

Competition: carvana-image-masking-challenge
Rank: #11
Source: https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40126

My solution is in two stages.

**Stage 1:  Predict rough outline of mask with 1/4 downscaled image**

 1. Scale down all images/masks to 320x480
 2. Train with default UNET.
 3. Predict mask for test set

**Stage 2: Train/Predict mask only using images around edge areas.**

This reduces the amount of data to process a lot, and I can use more complex network in reasonable time.

Details are:

 - Extract 256x256 tiles only along the edge of masks from stage 1 

 - Used deeper UNET - 6 downsample/upsample. But for each depth, Conv
   layers    with small number of layers than original UNET to avoid
   explosion of
       parameter numbers. Around 3million parameters are used.

 - During training, tiles are overlapped so that each edge pixel    are
   covered twice.

 - During inferencing, tiles are overlapped so that       each edge
   pixel are covered at least three times. So each pixel's    prediction
   is voted at least by 3 predictions(see attached image) UNET tends to    have    bad
   prediction around edge area, but this method reduced this problem.

 - Used standard augmentation(scale+-10%, flip left-right, brightness, contrast, pixel shift, etc) and basic dice_coeff as loss.
 - Used batch normalization, and to maximize the effectiveness of it,   
   the batch consists of random tiles from multiple images. I guess this
   helped network to converge much faster.

I tried several other segmentation networks, but this one showed best result with large margin(0.0002) compared to other models. Single model LB was 0.9970 and with ensemble of several folds from same model, it got to 0.9971. 

I think this pipeline works really well considering its simplicity, but I made a big mistake.
 - I didn't noticed that some test outputs from stage 1 were horribly wrong until today and didn't have chance to correct it.  :(
I should have found the issue early only if I had good visualization or simple sanity check logic.


The best lesson from this competition is that good visualization/analyzing tool is really important.


Oh, Heng CherKeng, thank you very much. I learned a lot from your posting!
