# 31rd Place Solution: CAM only.

Competition: hpa-single-cell-image-classification
Rank: #31
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/239048

Thanks Kaggle for hosting this incredible game, I really learned a lot from it. And thanks the Kagglers for the awesome ideas. This is my highest rank in my Kaggle trip so I decide to write the journey down.

# Summary

My final solution only use CAM(Class Activation Map) because other tricks does not work well for me. So it's a simple, e2e solution other than a complex pipeline. And it suprised me that the CAM only pipeline can give me a silver place.

I used Knowledge Distillation for the CAM layer becuase I think it will help the model to generate more slid CAM, well, it does.

I change the model downsampling size to 16(2**4) so for a 512 x 512 input, I will get a 32 x 32 CAM. And I use 3 layers output to get 3 CAMs at once.

The only working backbone for me is ResNest and all its variant (Thank You Zhang Hang).

# Training

The picture below describes the whole training pipeline:



##### CAM Extractor

The DRS and Dropout layer could force the model focusing on the whole image other than one or two cell. DRS can found here: https://github.com/qjadud1994/DRS



##### Training Details

Input: 512
Optimizer: Adam 
Loss: BCE
CV: 1 fold and then fine tune on whole dataset(5 fold is too expensive for me)
Augmentation: flip, random contrast
Model: ResNest and all the variant

# Inference

I finally used 5 models toghether(all from timm) to ensemble, all the model input is 512 x 512.

1. resnest50d_1s4x24d with DRS
2. resnest101e
3. resnest50d_1s4x24d
4. resnest50d_4s2x40d with DRS
5. resnest50d with DRS

TTA: flip both vertically and horizontally, for every spatial changed image, the contrast is changed too, the contrast is changed as 0.64, 0.88, 1.12, 1.36.

Cell Segment: original segmentator provided by the host.

Cell Scoring: Sum the pixel value of the CAM in the cell area, and divided by the nonzero pixel numbers in that cell.

**Thanks for reading.**
