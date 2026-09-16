# 8th Place Solution for the Open Problems – Single-Cell Perturbations Competition

Competition: open-problems-single-cell-perturbations
Rank: #8
Source: https://www.kaggle.com/c/open-problems-single-cell-perturbations/discussion/459618

Congrats to all the winners, and thank you to Kaggle for organizing such an interesting competition. And also thanks to the other kagglers who shared their ideas and notebooks.

# Context
- [Competition Overview](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview)
- [Competition Data](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data)

# Overview of the Approach

### Model
For model development, I designed and fine-tuned a simple neural network through a series of experiments, aiming to reduce both CV and LB scores. I used augment features from the [notebook - [3] OP2 - Feature Augment & Fragments of SMILES](https://www.kaggle.com/code/mehrankazeminia/3-op2-feature-augment-fragments-of-smiles?scriptVersionId=150423767&cellId=24) as model inputs.

Here is the model architecture:
```python
class SingleCellModel(nn.Module):
    def __init__(self, dim_size, mul_ratio, labels=18211):
        super().__init__()
        self.ce_layer = nn.Linear(labels, dim_size)
        self.sm_layer = nn.Linear(labels, dim_size)

        hidden_size = dim_size * 2
        self.fc1 = nn.Linear(hidden_size, hidden_size*mul_ratio)
        self.fc2 = nn.Linear(hidden_size*mul_ratio, hidden_size)
        
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_size, labels)
 
    def forward(self, cell_type, sm_name):
        x1 = self.act(self.ce_layer(cell_type))
        x2 = self.act(self.sm_layer(sm_name))

        x = torch.concat([x1, x2], dim=-1)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        x = self.out(x)
        return x
```

### Data Augmentation
The training process invloved a strategic approach to data augmentation. Initially, I employed only mean values for the cell_type and sm_name, respectively. Subsequently, I explored various statistical values such as median, min, max and quantiles. And I found out that median values significantly improves the LB score.

Moreover, I experimented with combinations of these features. I implemented 50% random selection between mean and median, and 25% random selection among mean, median, Q1 and Q2 for both cell_type and sm_name.  

### Validation Strategy
I used K-Fold cross-validation stratify on cell_type, trained 5, 10, 15, 20 splits. I reviewed that LB score increases with 10 and 15 splits.

# Details of the submission

The results of augmented models are summarized below, along with their respective LB scores.
1. median / 0.549
2. mean and median / 0.549
3. mean, median, Q1 and Q2 / 0.551

The final submission was a weighted average of these models by 0.35/0.35/0.3, which boosted up the LB score to 0.547. Since they had different prediction distributions with similiar LB scores, I believed that the ensemble would generalize well in the private.  

### Things that didn't work
- pseudo labels
- dropout
- normalization
- data selection (control, etc.)

# Sources

- [Learning single-cell perturbation responses using neural optimal transport](https://www.nature.com/articles/s41592-023-01969-x)
- https://www.kaggle.com/code/mehrankazeminia/3-op2-feature-augment-fragments-of-smiles
