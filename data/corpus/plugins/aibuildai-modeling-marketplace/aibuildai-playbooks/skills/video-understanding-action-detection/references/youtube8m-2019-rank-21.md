# Solution for 21st place and a PyTorch kernel

Competition: youtube8m-2019
Rank: #21
Source: https://www.kaggle.com/c/youtube8m-2019/discussion/112388

Out solution is pretty straightforward. We didn't use the train dataset at all. We trained a simple model (pooling + MLP) with a softmax classifier for 1000 classes. A single fold of this model resulted in 0.750 on public LB, ten folds resulted in 0.760 and a blend of 4 similar models resulted in 0.761, which is our final result.

The best model looks like this (exact values were found with hyperopt):
```
nn.AdaptiveAvgPool1d(1)    # average in time dimension

nn.Linear(1152, 2765)
nn.BatchNorm1d(width)
SwishActivation()

nn.Linear(2765, 1662)
nn.BatchNorm1d(width)
SwishActivation()

nn.Linear(1662, 1000)
```
I created a kernel with this model, and it achieves a public LB score of 0.753 in less than one hour, including submission generation: https://www.kaggle.com/artyomp/stronger-baseline/. This became possible since my teammate @tenich: 
1. has created a TensorFlow dataloader for PyTorch;
2. optimized submission generation from 5 hours to 5 minutes (check out `generate_submission`).

I used a slightly different approach regarding data loaders: I converted all data to np.arrays. This makes training a lot faster since 90% of validation data is not labeled. So, actually the size of the segment-labeled dataset is 2 Gb. 1 epoch takes less than a minute.

For the sake of speed, this kernel uses combo-approach. It only converts the validation data to np.array and uses TFDataset data loader for the final inference.

I hope this sheds some light on how challenges of this kind could be solved!
