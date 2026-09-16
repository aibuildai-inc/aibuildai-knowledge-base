# 8th place solution

Competition: hpa-single-cell-image-classification
Rank: #8
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238361

Since we are given image-levels labels and cell segmentations, we can train on image labels and predict cell labels directly.

Let's say during training we sample B images, within each image we sample M cells. Let's further assume input channels=4, input_size=256x256, feature_dim=2048. 

1. Input shape to CNNs -> BMx4x256x256
2. CNN feature extraction -> BMx2048x8x8
3. GAP -> BMx2048
4. Reshape and permute -> Bx2048xM
5. Another GAP -> Bx2048 
6. Last linear -> Bx18 (compute loss with image labels)

During inference, to predict cell labels, we remove step 4 and 5:
1. Input shape to CNNs -> BMx4x256x256
2. CNN feature extraction -> BMx2048x8x8
3. GAP -> BMx2048
6. Last linear -> BMx18 (cell label prediction completed)

This approach is partly inspired from the [lafoss concat tile pooling](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/146855) and the class activation mapping. 

Postprocessings that bring some improvement, ranked roughly by their importance:
1) Downscale border cells predictions.
2) Cell-level finetuning with OOF predictions in CV.
3) 0.9xcell predictions+0.1xaverage cell predictions within image, to make use of label correlations within a image. It should be better to train some independent image level models though [example](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/230940). Here I just reused cell predictions.
