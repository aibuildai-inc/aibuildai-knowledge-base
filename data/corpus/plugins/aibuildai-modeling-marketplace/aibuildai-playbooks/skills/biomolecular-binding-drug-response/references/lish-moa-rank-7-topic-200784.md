# 7th place solution

Competition: lish-moa
Rank: #7
Source: https://www.kaggle.com/c/lish-moa/discussion/200784

Thank you to the LISH Competition Organizers, the Kaggle Team, and everyone who shared notebooks and contributed to the discussions. 

TL;DR: Our final submission was a simple weighted average of an MLP, TabNet, and ResNet model.

## Dimensionality Reduction
We decided to do dimensionality reduction on the full test set. This meant we had to retrain the models when the private test set became available and so we couldn't do an inference-only submission. We used a mix of PCA, SVD, and ICA depending on the model.

## Early Stopping for Each Target
All of our models used early stopping for each target. The idea was to keep track of the best epoch for each of the targets separately. For example, this plot shows the log loss on a few of the targets.
[target loss]
If we had used a single epoch's weights for every target, we would have overfit some targets and underfit others.

## Validation
We tried a lot of variants of CV, but in the end, we went with MultilabelStratifiedKFold and did not use `train_drug.csv`. Our CV-LB alignment was still shaky and there was clearly a point at which further CV improvement degraded LB.  [@Chris Deotte's](https://www.kaggle.com/cdeotte) smart [CV scheme](https://www.kaggle.com/c/lish-moa/discussion/195195) was promising, but it did not solve this entirely for us.  Our LB scores with the smart CV scheme were a bit worse than with MultilabelStratifiedKFold, similar to what other teams noticed.

## Final Blend
Our final submission was a simple weighted average of `0.5*mlp + 0.3*tabnet + 0.2*resnet`. We did not use the public LB to tune these weights, and that probably helped prevent some overfitting.
| Model  | Public  | Private | Run Time |
| ------ | ------- | ------- | -------- |
| Blend  | 0.01817 | 0.01603 | 5883.6s  |
| MLP    | 0.01825 | 0.01613 |  816.6s  |
| TabNet | 0.01828 | 0.01614 | 1619.9s  |
| ResNet | 0.01859 | 0.01628 | 2562.4s  |

### MLP
Our MLP was very similar to those in public notebooks. We found that the PReLU activation worked better than ReLU or LeakyReLU.

### TabNet
We used the code from [@Optimo's](https://www.kaggle.com/optimo) excellent [TabNet Multitask Classifier](https://www.kaggle.com/optimo/tabnetmultitaskclassifier) [library](https://github.com/dreamquark-ai/tabnet/).  We used only the model code and not the helper functions from the library so that we could have more flexibility with the learning rate scheduler, loss function, etc.

### ResNet
Our model was based off of [@Demetry Pascal's](https://www.kaggle.com/demetrypascal) clever [2heads+deep resnets+pipeline+smoothing+transfer notebook](https://www.kaggle.com/demetrypascal/2heads-deep-resnets-pipeline-smoothing-transfer) and ported to PyTorch.
