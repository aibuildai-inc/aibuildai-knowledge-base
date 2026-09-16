# 7th place solution

Competition: leap-atmospheric-physics-ai-climsim
Rank: #7
Source: https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/524111

First of all, I would like to express my gratitude to the hosts @jerrylin96 and the Kaggle staff for organizing this interesting competition. It was a tough competition with issues of leaks, but the competition's task were very interesting and it was a great learning experience. I would also like to thank the community for sharing so much in the Discussions, including the discovery of leaks. And thank you to my team members @nomorevotch, @masatomatsui, @rheinmetall, I learned a lot from all of you.

# [Summary]
- We used various models, including LSTM, Transformer, Conv1D, and Squeezeformer. LSTM and Squeezeformer were particularly strong performers.
- Additional features based on domain knowledge contributed to improved accuracy.
- Training with MAE or SmoothL1Loss, followed by additional training with MSE, led to increased accuracy.
- For the ensemble, we used a weighted average with weights optimized by the Nelder-Mead method. (Public: 0.78560 / Private: 0.79080)
- In the ensemble, it was crucial to include a few strong single models rather than many models.
- It was important to speed up experimentation by not using HF's full data until the final week.

# [Ryota's Part]
### Data Preparation
- Use full low-res dataset from HF
- We sampled data at a 1/7 ratio from the period [0008-02, 0009-01], similar to the competition data, and used only 625,000 samples for validation.
- Use StandardScaler for scaling both input and target.
- Additional Features
    - Diff features calculated by taking the differences along the vertical axis
    - Diff features calculated by taking the differences of the aforementioned diff features
    - Relative humidity ratio
    - Pressure difference
    - Water vapor pressure
    - Ice rate
    - (lat, lon)
        - Due to concerns that this could be considered leakage, I finally did not use it, but it gave a slight improvement (~0.0002)
- Tried the following additional features calculated along the vertical axis, but they were ineffective
    - Moving statistics (mean, std, max, min, median)
    - Lag features
### Model
| model type | CV | Public | Private |
| --- | --- | --- | --- |
| Transformer + LSTM | 0.78734 | 0.78567 | 0.78058 |
| LSTM | 0.78794 | 0.78682 | 0.78120 |
| Conv1D | 0.78635 | 0.78301 | 0.77506 |



- Input / Output
    - Repeat the scaler features in the sequence direction, and the shape is (batch, 60, 25)
    - The output shape is (batch, 60, 14)
        - The scaler features are averaged across the entire sequence
- Get diff features
    - Calculate the aforementioned diff features in the forward method
    - The shape is (batch, 60, 86)
- Convolution Feature Extractor
    - Using 2 layers of convolution with Linear layers before and after
- Positional Embedding
    - Same as sinusoidal positional encoding but used as learnable parameters
- Transformer Encoder
    - PyTorch's Transformer Encoder
- Bi-LSTM Block
    - Each LSTM layer is followed by a Linear layer, with skip_connections applied to each layer, similar to a Transformer Block
- ResNet Block
    - Similar to ResNet, each block contains two convolutional layers with a skip connection to the input
    - In the latter 7 blocks of the Conv1D, an inception-like structure is used, applying a bottleneck structure and multiple parallel convolutional layers with different kernel sizes (1, 3, 5, 7).
    - Use SE-Block
- Head
    - 2 layers of Linear
- Activation
    - ELU for Conv1D
    - GELU for Transformer and LSTM
    - ReLU for Head
- Normalization
    - Batch Normalization for Conv1D
    - Layer Normalization for Transformer and LSTM
- No Dropout

### Loss
- MAE
    - MAE performed better than HuberLoss or MSELoss.
- Mask target columns where the weight is 0 or is included in ptend_q0002_[12, 26]
- Fine-Tuning by MSE
    - This trick consistently led to an improvement of about 0.002

### Training
- epoch
    - MAE : 13 epochs
    - MSE(Fine-Tuning) : 5 epochs
- optimizer
    - AdamW
        - lr=[5e-4, 5e-6]
        - weight_decay=0.01
- scheduler
    - Cosine schedule with warmup

### Post-processing
- Applied the post-processing described [here](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484) to ptend_q0002_[12, 26]
- As an additional post-processing, after calculating next_state_[q0002, q0003]_[all_levels] from the predictions, apply the above post-processing if the values are below threshold.
    - This led to an improvement of about 0.001 when using only the competition data, but there was a negligible improvement after using all the low-res data.

### Source Code
- All code is [here](https://github.com/nocchi1/kaggle-leap-7th-place-solution)

# [sqrt4kaido's part]
### Overview


### Validation
From the low-res data, I extracted 625,000 rows from the future period (February 2008 to January 2009) relative to the kaggle data and used them for validation.

### Feature Engineering
For features with sequences, we used the following:
```
"state_t",
"state_q0001",
"state_q0002",
"state_q0003",
"state_u",
"state_v",
```
For non-sequence features, we used all of them.
In addition to the data, the following are calculated:
- dp: Pressure difference
- RH: Relative humidity
- vp: Vapor pressure
- state_ice_rate: Ratio of ice in cloud water content (water + ice)
- ice_rate_diff: Difference between ice Ratio derived from temperature and state_ice_rate

After adding the above features, standard scaler is applied. Using max(1e-6, std) for the std.
Then, the following process is applied:

- Sequence features
Shaped into (60, num_feature) form. Diff and diff of diff features (both in negative and positive directions) are added.

- Non-sequence features
Repeated 60 times to match the sequence features.

In the end, we used 11*5 sequence features and 16 non-sequence features.


### Models

I used 1D sequence models.
- SqueezeFormer: Refer to [RNA 2nd solution](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460316)
- LSTM

Using SmoothL1Loss as the loss function. Loss is calculated only for the columns targeted in the test set.

### Post-processing
- [Replacement](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/499896)
"ptend_q0002_12 ~ ptend_q0002_28" columns was replaced with "state_q0002_12 ~ state_q0002_28".

- Adjustment to ensure that percentage values do not fall below 0
As state_q0002 and state_q0003 must always be non-negative, I adjusted ptend_q0002 and ptend_q0003 to ensure that the next time step's state_q0002 and state_q0003 would not fall below 0.

### Other Notes
- As a second stage, additional training with MSELoss boosts performance by about 0.002.
- When using low-res dataset, training is possible without loading all data into memory by using hdf5py.
- I had not used the full low-res dataset until the final week, conducting experiments using only Kaggle data. (Result: rank jump-up on the final day)

### Score
|model|public|private|
|---|---|---|
|squeezeformer|0.78511|0.78056| 
|LSTM|0.78094|0.77629|

# [e-toppo's part]
### Feature Engineering
In addition to the data, the following are calculated:
-   dp: Pressure difference
-   RH: Relative humidity (Reference: [**Climate-invariant machine learning](https://www.science.org/doi/10.1126/sciadv.adj7250))**
-   vp: Vapor pressure
-   state_ice_rate: Ratio of ice in cloud water content (water + ice)
-   ice_rate_diff: Difference between ice Ratio derived from temperature and state_ice_rate
-   q0005: q0002 + q0003
### Models & Training
-   Model: LSTM
-   Loss: Smooth L1
-   Auiliary Loss: ptend_RH
### Post-processing
-   Adjustment to ensure that percentage values do not fall below 0 As state_q0002 and state_q0003 must always be non-negative, I adjusted ptend_q0002 and ptend_q0003 to ensure that the next time step's state_q0002 and state_q0003 would not fall below 0.
-   Temperature Adjustment As state_q0003 must be 0 for temperatures above 274 degrees, ptend_0003 was adjusted to ensure this condition is met.

# [Rheinmetall's Part]
### Data
use all ClimSim_low-res in Hugging Face.
### Preprocessing
Apply StandardScaler to both features and targets.
### Input
- There are two types of features and targets, one with height dimension and the other with scalar quantity, respectively. Therefore, for both 556 dimensional features and 368 dimensional targets, we split them into sequence features and scalar features. 
- No additional input features are created.
### Validation
- After random shuffling, make the tail 625,000 as valid data.
### Model

- Sequence features are embedded in a linear layer and then input to LSTM, while scalar features are also embedded in a linear layer and then used as input to LSTM as hidden states. 
- The LSTM block has six layers, which was determined by a trade-off between model training time and accuracy.
### Outputs
- Since my model has two outputs, a sequence head and a scalar head, I reconstruct this in competition format, 368 dimensions.
### Loss function
- calculated based on MSEloss at each head, but multiplied by 0.1 for the scalar head loss. (to prioritize training on sequence heads)
### Post-processing
- Check all data, and if a non-negative column is negative, fill it with 0.

# [Ensemble]
- Ensemble method is weighted average of top 6 single models with weights optimized by the Nelder-Mead method.
- The Public best and Private best were the same submission. (Public : 0.78560 / Private : 0.79080)
- Including derivative models with lower single scores in the ensemble did not lead to improved accuracy. The key was to generate a strong single model
