# 5th place solution with code

Competition: asl-signs
Rank: #5
Source: https://www.kaggle.com/c/asl-signs/discussion/406491

Here is a quick overview of the 5th-place solution.

1.  **we applied various augmentations like flip, concatenation, etc**
1.1. By applying different augmentations, we can increase the cv by ~ 0.02 (0.76 -> 0.78)

2. **the model is only a transformer model based on the public kernels**
2.1. By increasing the number of parameters, the performance of a single model can be increased to around 0.8 (0.78->0.8) in public LB.
2.1.1. 3 layers of transformer with the embedding size 480.

3. **Preprocessing by mean and std of single sign sequence**
3.1. the preprocessing does affect the final performance. 
3.1.1. we tried different ways of calculating the mean and std and found out that using the mean and std of the single sign sequence results in better cv.

4. **Feature engineering like distances between points**
4.1. we selected and used around 106 points (as the public notebook by Heck).
4.2. distances withinpoints of hands/nose/eyes/... are calculated.

5. **some methods to prevent overfitting like awp, random mask of frames, ema, etc ...**

many thanks to my teammates  @qiaoshiji @zengzhaoyang

The source code for training models can be found here : https://github.com/zhouyuanzhe/kaggleasl5thplacesolution
