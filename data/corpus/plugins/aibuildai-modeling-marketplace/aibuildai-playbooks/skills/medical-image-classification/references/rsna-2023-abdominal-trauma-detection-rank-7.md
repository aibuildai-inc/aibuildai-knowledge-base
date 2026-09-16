# 7th Place Solution & Code

Competition: rsna-2023-abdominal-trauma-detection
Rank: #7
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447549

Thank you RSNA for hosting this great competition, which gave me a nice experience and  I believe this will start a wonderful journey on the Kaggel.

 I will briefly introduce the solution I used in this competition.

## Dataset

### sequence image data

1. My Solution is a 2.5D pipeline，it's necessary to process the sequence to a certain shapes, that is，**[T * 3, 512, 512]**  , every series will be **sampled** to a length, for example, T = 32. 

2. Then each independent slice image  be **croped** to include as much valid data as possible in the image. This can be achieved by counting effective pixels.

3. Finally reshape them to **[256, 384] **shape.

The visualization of cropping and reshaping results is as follows.




### sequence organ mask

Just use [the total segmentor model](https://pubs.rsna.org/doi/10.1148/ryai.230024) to generate segmentation results for all series. The bowel_mask = colon_mask + duodenum_mask + small_bowel_mask + esophagus_mask.

These masks will be used as mask ground truth, to assist with classification tasks.

## Models

**Backbone:** 

InternImage (base)     ->     out stride (8, 16, 32)

**neck:**

UnetPlusPlus      ->      out stride (4, 8)

**head:**

I think the head section is the most valuable and effective part of this scheme.

The bowel, liver, spleen,and kidneys all have specific shapes and positions，except for extravasation. So there is two heads for classification.

For the first head, I referred to the decoding idea of mask2Former which learned to predict a mask from a query, and using it as the attention of the decoder layer. This can help each query extract effective information for each organ.

For the extravasation head, using the image level label to assist feature learning and enable better classification.

The entire pipeline is as follows.



**loss:**

All cross entropy loss  are weighted according to the status of each organ in each patient. The organ weight consistent with the weight used for official verification.

## Post processing 

I simply averaged the results of different series of the same patient id.

## Ensemble
All ensembled models use the same model architecture, only using different sequence lengths(T=24/32/48) and different data folds.


### train code:
https://github.com/llreda/RSNA/tree/master

### inference code:
https://www.kaggle.com/code/hongx0615/rsna-2023-7th-place-solution-inference
