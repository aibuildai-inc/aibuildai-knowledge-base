# 3rd Place Solution for the Open Problems – Single-Cell Perturbations

Competition: open-problems-single-cell-perturbations
Rank: #3
Source: https://www.kaggle.com/c/open-problems-single-cell-perturbations/discussion/458750

# 1. Integration of Biological Knowledge
Generally, I treated this problem as a regression with 2 feature columns and 18211 targets, but I had tried to utilize SMILES sequences in neural network with LSTM unit. Both the sm_name and SMILES columns can be encoded exactly in the same way, so the sm_name column can be replaced with the SMILES column. Moreover, SMILES column is more informative, because every single character of the sequence can be encoded (not only a single value like in sm_name)  and the order of these characters provides extra information. Theoretically, In the worst case, the performance of the neural network using SMILES column instead of sn_name should be not worse than using original columns. Unfortunate, I have reached an unsatisfactory public score with this neural network and I stopped further research. 

# 2. Exploration of the problem
For simplicity, the analysis is done for the single best model without pseudolabeling. Since mrrmse metric is sensitive to outliers, a distribution of ranges for columns should be checked.

The majority of columns have a range of values in an interval (4, 50). Naturally, the columns with high range lead to high mae or mse, so in order to determine genes which are easy and hard to predict, the standardized (divided by std) colwise mse is applied. The table below shows the hardest and easiest genes for prediction.

The scheme of the applied cross validation.  Every fold contains one cell type chosen from NK cells, T cells CD4+, T cells CD8+, T regulatory cells and only sm_names being in public and private test was involved. The lowest value of this validation split corresponds to the lowest value on public and private dataset. In my opinion this is a reliable split and the perfect split depends on the model architecture, so every model can have its perfect validation split. 

The easiness of learning per cell types is shown below. 

The values of loss are different, because they are calculated in truncated space. T cells CD4+ and NK cells are learning well. T regulatory cells are harder for training. The T cells CD8 are weakly improving on validation dataset. This split uses about 25% of the dataset for validation, so I believe more reliable splits exist. My new proposition of split is: 

This split is similar to the previous one, but for each validation fold, randomly selected examples are moved to training. Let's check an impact of the new split for training. 

For each cell type, more data improved performance on this same, constant and small validation set. Metric is calculated on full dimension this time. The first value at 480 training examples corresponds to the previous split. Theoretically, decreasing the size of validation set to one example can lead to the best performance. Since it will be very similar to training on whole dataset, what is done finally.

#3. Model design
## Solution
The prediction system is two staged, so I publish two versions of the notebook. 
The first stage predicts pseudolabels. To be honest, if I stopped on this version, I would not be the third. 
The predicted pseudolabels on all test data (255 rows) are added to training in the second stage.
### Stage 1 preparing pseudolabels
The main part of this system is a neural network. Every neural network and its environment was optimized by optuna. Hyperparameters that have been optimized:
a dropout value, a number of neurons in particular layers, an output dimension of an embedding layer, a number of epochs, a learning rate, a batch size, a number of dimension of truncated singular value decomposition.
The optimization was done on custom 4-folds cross validation.  In order to avoid overfitting to cross validation by optuna I applied 2 repeats for every fold and took an average. Generally, the more, the better. The optuna's criterion was MRRMSE. 
Finally, 7 models were ensembled. Optuna was applied again to determine best weights of linear combination. The prediction of test set is the pseudolabels now and will be used in second stage.
### Stage 2 retraining with pseudolabels
The pseudolabels (255 rows) were added to the training dataset. I applied 20 models with optimized parameters in different experiments for a model diversity.
Optuna selected optimal weights for the linear combination of the prediction again.
Models had high variance, so every model was trained 10 times on all dataset and the median of prediction is taken as a final prediction.  The prediction was additionally clipped to colwise min and max. 

**History of improvements:**
1. a replacing onehot encoding with an embedding layer
2. a replacing MAE loss with MRRMSE loss
3. an ensembing of models with mean
4. a dimension reduction with truncated singular value decomposition
5. an ensembling of models with weighted mean
6. using pseudolabeling
7. using pseudolabeling and ensembling of 20 models and weighted mean. 

**What did not work for me**:
- a label normalization, standardization
- a chained regression
- a denoising dataset
- a removal of outliers
- an adding noise to labels
- a training on selected easy / hard to predict columns
- a huber loss.

# 4. Robustness
I have tested 3 types of the robustness: increasing dataset size, adding noise to labels and inputs. Adding the noise to inputs failed totally, it is logical for me, because the nominal and categorical values are changed and are behaving like the continues values, what is not beneficial.
Let's see the performance on 40%, 50%,..., 100% of training dataset. It started from 40%, because singular values decomposition is limited by number of examples.

The experiment was 5 times repeated, so the interval of uncertainty is visible. More data improves the performance significantly. 
The last test of robustness is adding noise to the labels. The random gaussian (a distribution with 0 mean and scale * std) noise was added to the labels.

Adding some noise (0.01 * std) can even improve the model's performance.  Generally, the model is robust to the noise. 

# 5. Documentation & code style
The code on GitHub is documented. 

# 6. Reproducibility
GitHub code:
[repo](https://github.com/okon2000/single_cell_perturbations)
Notebook. The version 264 is first a stage and 266 the second one:
[notebook](https://www.kaggle.com/code/jankowalski2000/3rd-place-solution).

The code runs in approximately 1 hour using CPU Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz and 8GB RAM.
