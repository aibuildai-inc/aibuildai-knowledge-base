# 2nd Place Solution

Competition: tabular-playground-series-apr-2022
Rank: #2
Source: https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322257

First, I would like to thank Kaggle for organizing such interesting and leak-free competitions.

My final solution was a stacking of 40 models, each of which scoring 0.96+ on both local CV and LB. I don't think there something special to share in my stacking solution. However, one single model of mine has achieved 0.987 (LB), 0.989(Private LB), and 0.963 (CV) which I think might be the best single model in this competition and can alone land the 6th place in this competition. 

### Feature Engineering
I didn't use new features compared to the public one, expect for grouping *sensor_02* readings by the engineered **count** feature, which I don't think has played a major role in the final score.

### Cross Validation
I used the StratifiedKFold method stratified on the label after reshaping the Numpy form of the dataframe into the shape (-1, 60, train.shape[-1]). In this case you don't need to care for the groups as each index now is a full group by itself.

### Best Single Model
My best single model is based on 4x2D-CNN + GRUs + GMP. This model can be found [here](https://www.kaggle.com/code/azzamradman/tps04-best-single-model-0-989), and the submission can be found [here](https://www.kaggle.com/code/azzamradman/best-single-model-submission-0-989).

During training, I noticed the variation between each fold's output in terms of AUC. Hence, each fold was run 3 times, and the best of these three iterations was used for the final best single model. The remaining 2 iterations were also then exploited as meta-features for the meta-learner.

### Stacking
I saved the meta-features (OOFs) of all my good models (0.96+) and built a final LGBM model on top of them. This step enable getting a boost of 0.00130. My final CV AUC score is 0.99. [Here](https://www.kaggle.com/azzamradman/tps-04-blending) you can find the final stacking.

### Things didn't work well
1. Denoising AutoEncoder (DAE): I treated each sequence as an image using Conv2D and MaxPooling2D layers to compress it and used GlobalAveragePooling2D to flatten each sequence in the latent space. Conv2DTranspose + Conv2D were used to re-construct the original image (seuqence) from its representation in the latent space. DAE seems to be working with LSTM as in @davidedward12's solution.
2. Transformer Encoder: its training was tricky and the results were unstable.
