# 14th place solution NN (surviving the big shakeup)

Competition: playground-series-s3e5
Rank: #14
Source: https://www.kaggle.com/c/playground-series-s3e5/discussion/386627

**Quick intro:**
Congratulations to the winners of this competition! it was challenging to maintain the position, especially when the metric would not forgive a bad misclassification. I wish it was a topic of the top-3 solution but finishing in 14th place was no small feat here. This was the one there I survived among the few. 

**CFG:**

```python
config = {
            'model_config': {
                'act_fn': F.mish,
                'learning_rate': 1e-3,  # 1e-3,
                'num_classes': 6,
                'hidden_sizes': [1024, 512, 128, 256],
                'drop_out': .50,  # 0.3,
                'norm_last_layer': True,
                'weight_decay': 1e-5,
            },
            'seed': s,
            'num_folds': 6,  
            'batch_size': 128,
            'num_epochs': 13,
            'plateau_factor': .5,
            'plateau_patience': 3,
            'exp_num': 1,
            'combined_data': True
        }

```
**Feature Engineering:**

```python
df['alcohol_density'] = df['alcohol'] * df['density']
df['sulphate/density'] = df['sulphates'] / df['density']
df['sulphate/alcohol'] = df['sulphates'] / df['alcohol']
df['pH_round1'] = df['pH'].round(1)
df['log1p_residual_sugar'] = np.log1p(df['residual_sugar'])
df['citric_acid_per_alcohol'] = df['citric_acid'] / df['alcohol']
conditions = (df['citric_acid'].eq(0), df['citric_acid'].eq(.49))
df['alcohol_mean_group_by_pH'] = df.groupby('pH_round1')['alcohol'].transform('mean')
```

**Model:**
`nn.CrossEntropyLoss(weight=torch.tensor([1.10,  1.5,  1.,  1.,  1.5, 1.5]), reduction='sum')`
```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                  [-1, 512]           8,192
       BatchNorm1d-2                  [-1, 512]           1,024
           Dropout-3                  [-1, 512]               0
            Linear-4                  [-1, 256]         131,072
       BatchNorm1d-5                  [-1, 256]             512
           Dropout-6                  [-1, 256]               0
            Linear-7                  [-1, 128]          32,768
       BatchNorm1d-8                  [-1, 128]             256
           Dropout-9                  [-1, 128]               0
           Linear-10                    [-1, 5]             645
================================================================
```
