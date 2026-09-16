# 16th place solution [0.988 private LB]

Competition: recursion-cellular-image-classification
Rank: #16
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110361



First, I would like to thank Recursion Pharmaceuticals and Kaggle for organizing such an interesting competition. 
Then, I deeply appreciate [nosound](https://www.kaggle.com/zaharch) and [Giulia Savorgnan](https://www.kaggle.com/giuliasavorgnan) for reporting the plate leak. 

My solution is pretty simple.
The overview is shown in the figure above. (**SORRY for my messy handwriting..**)

## Setup
- I used cloud instances with some V100s.
- PyTorch

## Data
- 6 channel, 512x512 input
- Contrast limited adaptive histogram equalization(CLAHE) is applied to some models
- Typical augmentation

## Model
- Basically cosFace with various backbones
- RAdam optimizer and cyclic learning rate
- Each site is treated separately
- 2 stage training from [this discussion](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/100414#latest-634062)
- Cross validation in 2nd stage training

## Prediction
- No```model.eval()``` (this is due to difference between each experiment)
- Test time augmentation(TTA) is carried out 8 times for each image
- Predictions from both sites are averaged
- Soft voting

## Post processing 
- Raw prediction is corrected in the same way [this kernel](https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak) does
- Hungarian algorithm is used to remove duplicated prediction in each plate 

## What didn’t work
- Mixup augmentation
- Control image ( I tried a two head model in which experiment image features are subtracted by control image features, but it performed worse)

I really wanted to try pseudo labeling, which was likely to boost the score, but I didn’t manage to do that due to lack of time.

**Finally, congratulations to the winners!**


# 
P.S.
I work at [Aillis Inc.](https://aillis.jp), a Japanese medical device startup developing advanced diagnosis device using throat images. We are hiring! Please contact me if you are interested.
