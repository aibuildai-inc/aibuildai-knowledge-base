# 31st place solution - single fold 512x512 UNet EfficientNetB4

Competition: siim-acr-pneumothorax-segmentation
Rank: #31
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107795

Congratulations to all of the winners and participants.

Here is my solution:

**Preprocessing**
Histogram equalization (CLAHE) with kernel size 32x32 on original image size 1024x1024 - (this kernel size gives more details than bigger sizes).

**Model training**
- UNet with EfficientNetB4 encoder
- Single stratified fold. Train - 83.34%, validation 16.66%.
- Adam
- BCE + Dice loss
- 100 epochs
- not aggressive augmentations with HFlip and low brightness changes (to keep histogram eq. work)
- Input image size 512x512, Batch size: 4
- Gradient accumulation - 4 batches
- LR: Cosine annealing - 4 cycles, from 1e-3 to 1e-6
- Stochastic Weight Averaging - 4 epochs on the lowest LR

**Prediction**
- HFilp TTA.
- Two-step thresholding:
    1. Find the threshold which fits for the classification task (reduce false positive), every time it is somewhere between 0.95-0.99. 
    2. Find the threshold, which fits for the segmentation task - usually between 0.01-0.50

So to the predicted sigmoid map (after HFlip TTA) was applied threshold with high value. All masks with less than 2% detected pixels marked as an empty mask. If there are more than 2% detected pixels, I apply the 2nd threshold with a lower value to produce a final mask.
