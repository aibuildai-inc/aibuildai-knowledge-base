# Public&Private 37th solution [Team CM, T0m Part]

Competition: commonlitreadabilityprize
Rank: #37
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/260939

Firstly, thanks to kaggle and hosts for an interesting competition, and to my teammate @teeyee314 , @nazarov , and @leecming . I learned a lot from them during this competition.
Since this is my first NLP competition, I started by reading BERT paper and thought a lot about how to train BERT stable and how to design the regression model.

So, I'd like to discuss about what I've done to achieve the above.

The followings are main idea of my part.
- Binary Pretraining using competition data.
- High Probability Mixout (at the same time, Dropout=0 in bert layer)
- Weighted Loss by using Standard_error
- Custom Aggregation and Head for each hidden layer

# Binary Pre-training
I did an additional pre-training that was inspired by the roots of the competition target. They are actually Bradley-Terry values and from pairs of excerpts.
So, I first designed binary-classification model, and then use the weights for main regression task.



Note:
- The bert's weights are shared.
- In train phase, pairs are randomly sampled.
- In validation phase, the pairs are should be a fixed one.
- There are many sampling strategy(ex, sample the pairs whose target difference is greater than a threshold), but random is the best.

The pre-training weight boost my CV and LB by 0.002 ~ 0.01.

# Weighted Loss
To make training more stable, I used the standard_error for calculating the training loss.
the details are below,
1, scaling standard_error range {0. ~ 0.5}
2, calculating MSELoss
3, Weighting the each loss by using scaled standard_error
4, Mean and Backward
```python
# pytorch brief example code
# scaling
values = train["standard_error"].values
train["standard_error"] = 0.5 * (values - np.min(values)) / (np.max(values) - np.min(values))
~~~
# training
criterion = nn.MSELoss(reduction='none')
~~~
loss = (criterion(output.squeeze(), labels) * (1 - errors)).mean()
loss.backward()
```

# Custom Aggregation and Head for each hidden layer
I added the regression head for each bert hidden layers, and then (learnable) weighted sum for predicting final target value.
Each aggregation and head weights are not shared. 



```python
class AggregationLayer(nn.Module):
    def __init__(self, in_size, max_length):
        super().__init__()
        weights_init = torch.zeros(max_length).float()
        self.seq_weights = torch.nn.Parameter(weights_init)
        weights_init = torch.zeros(in_size).float()
        weights_init.data[1:] = -1
        self.dim_weights = torch.nn.Parameter(weights_init)

    def forward(self, x):
        w_dim = torch.softmax(self.dim_weights, dim=0)
        w_seq = torch.softmax(self.seq_weights, dim=0)
        x_dim = (x * w_dim).sum(1)
        x_seq = (x * w_seq.unsqueeze(-1).expand(x.shape).float()).sum(1)
        z = (x_dim + x_seq) / 2
        return z
```


# Other tips
- add <unk> token randomly improved CV and LB (p=0.05)
- Strong Mixout improved CV and LB (p=0.4) (this may be an effect of the dropout being zero.)
- max_length = 300

# Model
With the above implementation, I achieved CV=0.4723 and LB=0.460 (roberta-large)

# After Team Merge
I made a lot of models with pseudo labeling (external data).
