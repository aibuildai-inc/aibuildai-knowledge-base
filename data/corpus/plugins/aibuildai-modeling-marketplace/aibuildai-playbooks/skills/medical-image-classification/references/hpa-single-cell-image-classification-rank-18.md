# 18th place solution private 0.515 public 0.514

Competition: hpa-single-cell-image-classification
Rank: #18
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238385

Thanks to kaggle and the organizers for putting together a competition on weakly supervised learning approach. learned a lot from this experience.

## solution overview
since this is a weakly supervised learning task and HPA cell segment model was able to give instance mask, i chose to use Multiple instance learning method here. this made sense as the image level labels are not precise and through discussion it was clear that there could be cells that are negative class but not mentioned on the image level label. 

## stage 0 model
Here i used Effb4 and resnest50 as the CNN backbone. the training setup consist of extracting individual cells from an image and selecting 16 cells (max, else resampling to match 16 count) at random to be used as Bag of images. the target is same as image level label. here i use 16 cells just to make sure i have at least some instance of cells corresponding to the label. (in stage 0 we cant be sure on cell labels) 

CNN Feature Extractor : Effb4, resnest 50
cell used per image : 16
Pooling layer (per cell): GAP
Attention pooling layer (Per image, 16 cells)
image level label
Focal loss
Time : roughly 1hr per epoch 

The magic of this approach is the attention pooling layer which weights the 16 cell feature embedding by providing importance score based on the contribution of individual cells to the image level label. this enabled me to train a model that can find the appropriate cell level label by looking at the image level label. 

## stage 1 model
after training stage 0 model, now we have a model that is capable of providing cell level protein score that can be submitted to the competition. along with this we also have a well trained attention layer that can serve as a module to score individual cells. this can be used as a pseudo label generator. i used this approach to find the missing cell labels in a image and append this to the image level label. this helped with finding images which consist of Negative classes but not labeled correctly. This also helped with reducing the cell count per image as now we are more sure about the class of cells present on the image.

CNN Feature Extractor : Effb4, resnest 50
cell used per image : 8
Pooling layer (per cell): Transformer based attention pooling
Attention pooling layer (Per image, 8 cells)
image level label + pseudo label
Focal loss
Time : roughly 20min per epoch 

we can clearly see how the training time dropped because of reducing the no of cells.  

## final inference 
My final inference consist of  5 model where 3 stage 0 and 2 stage 1 models were used. i did use 4 TTA to make my model prediction robust

i am happy that my public and private score for this submission did not change a lot Private 0.515 public 0.514. i did suffer from accuracy drop due to GPU hardware change. i trained my model on volta architecture GPU and Kaggle uses P100 (pascal). i got a scored of 0.524 on public dataset by generating the submission file from volta GPU. this was a surprise for me. i did try to calculate the mean of difference between the GPU submission files and used it to correct my final submission score. it did help me a bit but i lost a lot of score just because of Hardware difference. 

## final thoughts
overall i got a good experience with working on Weakly supervised learning problem. Thanks again
