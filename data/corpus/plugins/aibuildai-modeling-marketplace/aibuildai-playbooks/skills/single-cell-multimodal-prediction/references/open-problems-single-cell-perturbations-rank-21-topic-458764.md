# SCP 21st Solution

Competition: open-problems-single-cell-perturbations
Rank: #21
Source: https://www.kaggle.com/c/open-problems-single-cell-perturbations/discussion/458764

**Feature Engineering**

Broadly two different sets of features were used for different models used in the final ensemble:

Feature Set 1
1. Cell type - one-hot encoded
2. SMILES - converted to 2048 bit vectors using RDKit Morgan fingerprints
3. Drug properties: for each SMILE, log P and log of Molar Refractivity, standard scaled
4. Control - whether or not drug is a control (1) or not (0)

Feature Set 2
1. Same features as in Feature Set 1
2. Average SVD embedding values for drug effects on all cell types AND average SVD embedding values for cell response to various drugs
3. Exclusive of 2), or usage of average SVD embedding values for log fold-change for drug effects on all cell types AND average SVD emedding values for log fold-change for cell response to various drugs (either use 2 or 3)

Note that the number of SVD singular values to keep was chosen according to the Gavish-Donohoe (GD) SVD hard threshold method: https://arxiv.org/abs/1305.5870

Target
1. GD SVD criterion used for singular value cutoff for low dimensional modes to keep
2. SVD applied to target matrix and the singular value cutoff determined according to the Gavish-Donohoe threshold
3. All models were trained against the SVD embedding of the original target, and model predictions were transformed back using the transpose of the V matrix

**Model Architectures and Data Upsampling**

**Model 1:** 

Simple direct regression on SVD embedding targets
1. 8 layers Dense feed-forward neural netwok (5128 neurons per layer), output layer 114 neurons 
2. SELU activation each layer except for output layer (no activation, linear regression output)
3. Output is in the SVD embedding space (114 columns)
4. Loss: MAE or Pseudo-huber
5. Epochs: 800
6. Batch size: 16
7. Cosine training schedule with warm restart every 200 epochs (alpha = 0.01, t_mul = 1.0, m_mul = 0.9)
8. Stochastic weight averaging (SWA): SWA start from epoch 2
9. Predictions for 18211 genes:
    1. Let output be the predicted SVD embedding
    2. Take predicted SVD embedding and multiply by transpose of V matrix from SVD to get back to original 18211 representation
    3. Number of singular values to keep chosen according to Gavish-Donohoe threshold (see above)

**Model 2:**

Same architecture as Model 1 however sample weights introduced to loss function.

Sample weight scheme:

1. From training set filter out drug-cell pairs where B cells / myeloid cells were exposed to same compounds
2. From 1), exposure to same set of compounds but observed difference in target ( - log10(p_val) * sign(LFC)) should be attributable to cellular difference
3. For each cell type not B cells / myeloid cells calculate a notion of "distance" from the filtered and observed targets for exposure under same drugs using a distance metric of choice, e.g. Frobenius norm of difference of target matrices
4. For each cell type not B cells / myeloid cells average out this "distance" metric calculated in 3) and then subtract from 1 i.e. distance to B cell or myeloid cell would be 0 s.t. one minus this amount would give each cell type of prediction interest a score of 1, whilst cell types further away gets a lower score
5. Divide each cell type by the minimum score of the 6 cell types as calculated in step 4), and use this number as a weight for each row in training based on cell type used in the experiment
6. Model is trained on this weighted loss inclusive of each row's weight

**Model 3:**

Skip connections architecture:

1. 8 or 9 Dense layers
2. Skip connections:
    1. Input dimension: 2056
    2. Concatenate layer: Input concatenated with layer 2 pre-activation output (3072 neurons) leading to 5128 output dimension 
    (3072+2056) before feeding into SELU activation layer
    3. Additive skip connections: SELU output of concatenate layer (5128) + pre-activation output of layer 4 (5128), SELU
    output of layer 4 + pre-activation output of layer 6 (5128), SELU output of layer 6 + pre-activation output of layer 8
    (5128 / used where network has 9 hidden layers)
3. Other details similar to Model 1

**Model 4:**

Model 1 architecture but using training error to identify hard to predict drug-cell pairs for upsampling. Upsampling was done by identifying index of training samples (rows) which were at or below at certain training error threshold and then amplified by making a new copies (integer multiples) of these rows to be concatenated to original training set. 

The thinking here was that since the problem for predicting interactions for B / myeloid cells is potentially underspecified and to be extrapolated from observed interactions of other cells, the drug-cell pairs that have high row-wise accuracy or low MAE (or other regression metric) are not as important and performance on these rows can be sacrificed for better performance on the rows in training which have low row-wise accuracy or low MAE (or other regression metric). The amplified set was also manually checked for inclusion of the small number of B / myeloid cell observations in training.

Broadly three types of this upsampling procedure were used with various models

**Upsampling procedure 1:** Regression based row-wise metric (MAE) for determining cut-off threshold

1. Simpler smaller neural network trained for 200 epochs on original training set
2. Row-wise MAE computed for each sample
3. Take median of 614 row-wise MAE metrics
4. Take a positive multiple of this median (e.g. 3x or 15x) to select the base set of training rows to be upsampled
5. Make K times more (e.g. 7x) copies of the training subset in 4) and concatenate to original training set
6. Re-train larger model (could be any model architecture) on this upsampled training set

**Upsampling procedure 2:** Sign classification using logistic loss for determining cut-off threshold

The thinking behind this approach is that sign may be important to get right as an individual prediction where the magnitude (-log10(p_val)) is correct but where sign is not is very consequential for RWRMSE metric.

1. Same procedure as in prior upsampling procedure, except neural network with regression output is trained against the sign of
   the log fold-change (i.e. target matrix is composed of +1/-1)

    Logistic Loss = (1/n) * Sum(i from 1 to n) L(y, t) where
    L(y, t) = ln(1 + exp(-y * t))
    
    With t being in {-1, +1} i.e. the sign of the log fold change

2. Row-wise accuracy (%) is computed on the training set
3. Choose a cutoff below which the training rows are to be upsampled. I used arbitrary cutoffs such as 75% or accuracy cutoffs
   3 standard deviations below the mean row-wise accuracy
4. Repeat upsampling procedure as in the previous procedure amplifying this subset an integer number of times and retrain a
   larger model on this exapnded training set
   
**Upsampling procedure 3:** Sign classification but focussed on rows with bad sign classification for small p-values

Small p-values (e.g. less than 0.1) leads to large magnitudes when -log10 transformed, so intuition is get sign more correct for these as a bad sign classification flips these magnitudes to other side of real number line. 

Similar procedure to upsampling procedure, however we calculate accuracy only on subset of genes for each drug-cell pair where p-values are below a chosen threshold. Once these row-wise accuracy figures are computed, the same process as in the prior sign upsampling procedure is used to upsample a subset for retraining.

**Model 5:**

Triple regression head model with upsampling procedure and contrastive loss. The idea behind this architecture is to have share layers (e.g. 5 layers) between 3 different regression outputs. A "contrastive" loss (see below) was used to incentivise each regression head to learn a different hypothesis to the other 2 heads. This model architecture was mostly trained with sign upsampling procedure 2 as described in Model 4.

Architecture:

1. Shared weight layers: 5 Dense layers
2. Activation: SELU for shared layers and regression heads, linear activation for regression outputs
3. 3 regression heads: [3072, 2048, 1024] neurons before output layer for training y_train SVD embeddings
4. Contrastive loss: Sum(head 1 to 3) of regression loss for each head + contrast_weight * average_pairwise_dissimilarity
   
   K = n_head choose 2
   
   Average_Pairwise_Dissimilarity = 1/K * Sum(i from 1 to K) (Average Row-wise Cosine Similarity + 1.)
   
   If two non-zero vectors are exactly opposite, row-wise cosine similarity evaluates to -1. If they are exactly the same, we 
   get +1 and if they are orthogonal we get 0. Adding 1 to the average row-wise cosine similarity ensures the minimization
   objective goes to 0 (instead of -1).
   
   Contrastive loss essentially balances between each regression driving down bias but also learning distinctive hypotheses
   from the data. The amount of contrast between the heads is controlled by the contrast_weight
   
5. Each regression heads' output is multiplied by the transpose of the V matrix from SVD to get back predictions for original
   18211 genes. 
   
6. Some submissions used the best head's predictions as determined by training error. Other predictions ensembled the 3 heads'
   predictions by equal or training loss derived weights (lower loss -> higher weight)
  
**Final Submission**

The two final submissions were LB RWRMSE weighted ensembles of the 16 best and 80 best submissions.

For each submission I took the LB RWRMSE error, cubed them and subtracted from 1. to derive a score. These scores were then normalized against each other for the final weighted addition of the submissions.\

**Code**

https://github.com/maxleverage/kaggle-scp
