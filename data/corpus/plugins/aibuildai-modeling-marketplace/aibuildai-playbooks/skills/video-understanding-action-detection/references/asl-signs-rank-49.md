# 49th place silver solution

Competition: asl-signs
Rank: #49
Source: https://www.kaggle.com/c/asl-signs/discussion/406426

Thank you Kaggle and Pop sign for hosting this competition. I enjoyed working on this problem and learned quite a bit from really good [discussion posts](https://www.kaggle.com/competitions/asl-signs/discussion/391265)  and [code](https://www.kaggle.com/datasets/hengck23/asl-demo) shared - Thank you @hengck23 


- Pytorch 2.0  (pure pytorch no training library)→ ONNX → TFLITE
- Max length i.e. number of frames per video = 256 (Center crop)
- Normalization by LIP, SPOSE, Lhand and Rhand (total 90 kepyoints)
- Flip augmentation (p=0.5)
- Distance, motion and acceleration features  (Total 1050)
- Model almost same as shared [here](https://www.kaggle.com/code/hengck23/lb-0-73-single-fold-transformer-architecture])  - with positional encoding 16, added layernorm in FeedForwardNetwork and 2 fully connected layers with batchnorm, CLS token + mean pooling for final fully connected layer
- Loss function cross entropy with label smoothing = 0.75
- Epochs 50 - dropout 0.0 for 15 epochs, dropout 0.4 from 15-35 epochs, dropout 0.2 for the remaining
- FP16 quantization reduced model size significantly - final size ~26MB and ~60 mins for 5 folds

Things that didn’t work 

- External data - It improved by 3-4 ranks but decided to not use it , too messy data and no commercial license
- Adding more features, reducing max length, removing data with too high/low number of frames
- SWA - couldn’t get it to work
- Augment Affine improved CV and private LB but not public LB.
- Various hyper parameters like # of encoder blocks, embedding dimension
- Arcface Loss - Based on this [post](https://www.kaggle.com/competitions/asl-signs/discussion/406301)  it seems Arcface loss + label smoothing didn’t work . Should have tried just Arcface and no label smoothing.
- Also tried to convert TF model [here](https://www.kaggle.com/code/markwijkhuizen/gislr-tf-data-processing-transformer-training) to PyTorch

Based on private LB scores it seems that data in private LB is much less noisy than training set. 

----

My code here - https://github.com/rashmibanthia/ASL
