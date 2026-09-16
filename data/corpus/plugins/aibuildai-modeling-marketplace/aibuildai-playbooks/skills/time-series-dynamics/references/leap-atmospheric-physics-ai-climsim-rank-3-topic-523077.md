# 3rd place solution

Competition: leap-atmospheric-physics-ai-climsim
Rank: #3
Source: https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523077

Everyone, thank you for your hard work on the competition.


@jerrylin  Thank you for organizing the competition. I know it must have been tough, but I believe we were able to get this far thanks to your sincere dedication up to the final check.
Congratulations to everyone who ranked high and get good results. It was enjoyable to compete alongside you, and you all served as great motivation.
To my teammates @bamps53  @kmat2019 , it was fun and I learned a lot. Thank you very much.
Also I am happy to become Grandmaster at this competition. Also @kmat2019 became grandmaster. Congrats!!


## Model Summary
### Overall Summary

[Overall pipeline].png?generation=1722316979148053&alt=media)

Each team member built their own neural network models. 

After obtaining the Camaro model's predictions, we created several features to input into GBDT regressors. These regressors refined the model's predictions. Although this second stage could be applied to other models, we only applied it to the Camaro model because the Pao and Kmat models lacked a validation dataset.

The final prediction was calculated as a weighted average of these predictions.

### Pao Part

#### Overview
- **Model:** 1d CNN + Transformer + LSTM
- **Feature:** Original and relative humidity with those sequence 1st derivative and 2nd derivative at heights (diff and diff-diff)
- **Auxiliary Loss:** Predicting the difference between adjacent vertical levels
- **Row-less full training**

#### Model Architecture

[Pao model]

- **Input layer:** Linear transformation with bias at each feature
  - Example: `feature1 = feature1_original * a1 + b1` (a1 and b1 are trainable parameters)
- **Feature Concatenation:** Concatenate scalar features to each height sequence feature after each dense layer
  - Example: `features_level0 = concat([seq_features_level0, dense_level0(scalar_features)])`
- **Positional Encoding:** Adding embedding per height to the hidden layer
- **Model Blocks:**
  - **Residual block:** 1dCNN (Conv1d + BN + GELU) * 2 + Transformer
    - Conv1d: kernel size = 5
    - Transformer: n_head = 8, n_layers = 1
  - **LSTM:** Bidirectional 2 layers
  - **MLP:** Simple MLP with (Linear and GELU with no dropout)



#### Training
- **Auxiliary Loss:** Predicting the difference between adjacent vertical levels (similar to Camaro part)
- **Loss:** Huber Loss using EMA
- **Learning rate:** Cosine annealing 1e-3 to 1e-5, 10 epochs
- **Optimizer:** AdamW

#### Others
- **Dataset:** WebDataset for LowRes full-training
- **Normalization:** 
  - **Feature:** `(input - mean(input)) / std(input)`
  - **Target:** Multiply old_sample_submission weight
- **Postprocess:** Replace predictions for `ptend_q0002_0-27` with `-1 * input / 1200`
- **Ensemble:** 3 models with variations in dropout, hidden size, and Huber Loss delta

### Camaro Part

#### Overview
- **Full Dataset and Long Training:** Training on the full dataset
- **Model Architecture:** Combination of CNN and Transformer or Transformer-only using the CLIP Encoder
- **Auxiliary Loss:** Predicting the difference between adjacent vertical levels

#### Dataset and Preprocessing
- **HuggingFace Dataset:** Full dataset for training
- **Feature Engineering:** Added saturation vapor pressure as a feature
- **Normalization:** Pre-computed using Kaggle train and test datasets
- **WebDataset:** Efficient loading for faster training

#### Model
- **Architecture:** Combination of CNN and Transformer or Transformer-only with heavy embedding and head layers

#### Training
- **Target Transformation:** Dividing by old sample submission weights and subtracting the mean
- **Auxiliary Loss:** Incorporating information about derivatives and second derivatives
- **Loss Function:** Huber loss with a delta of 2.0

#### Post-Processing
- Replace predictions for `ptend_q0002_0-26` with `-1 * input / 1200`

#### Results
| exp | bs  | arch               | dim | loss   | Public LB | Private LB | Public LB (2nd stage) | Private LB (2nd stage) |
|-----|-----|--------------------|-----|--------|-----------|------------|-----------------------|------------------------|
| 1   | 256 | ConvTransformer    | 256 | Huber2 | 0.78468   | 0.78176    | 0.78504               | 0.78199                |
| 2   | 1024| ConvTransformer    | 384 | Huber2 | 0.78418   | 0.78131    | 0.78451               | 0.78159                |
| 3   | 512 | Transformer n_layer=8 | 256 | Huber2 | 0.78545   | 0.78154    | 0.78572               | 0.78154                |
| 4   | 768 | ConvTransformer x 2| 256 | Huber4 | 0.78395   | 0.78122    | 0.78427               | 0.78141                |
| 5   | 768 | Transformer n_layer=6 | 256 | Huber8 | 0.78245   | 0.77951    | 0.78255               | 0.77957                |
| **Ensemble (1+2+3+4)** |     |                    |     |        |           |            | 0.79025               | 0.78694                |
| **Ensemble (1+2+3+4+5)** |     |                    |     |        |           |            | 0.78998               | 0.78685                | 0.79023                | 0.78703                |

### Kmat Part

#### Overview
As shown in Fig.6-1, Kmat part consists of:
- Add Features
- Normalize by averages and standards
- 1D CNN model to predict climate
- Postprocess (some predictions are replaced by `-input/1200`)

#### Feature Engineering
- Diff features from 1D data: `x[z] - x[z-1]`
- Relative humidity-related features such as dew_point and vapor_pressure / saturation_pressure

#### Normalization
- **Inputs:**
  1. `(x - x_mean(axis=0)) / x_std(axis=0)`
  2. `(x - x_mean(axis=(0,1))) / x_std(axis=(0,1))`
  3. `(log_x - log_x_mean(axis=(0,1))) / log_x_std(axis=(0,1))`
- **Targets:**
  - `(x - x_mean(axis=0)) / x_std(axis=0)`

#### Model Architecture
- **Core Architecture:** FiLM 1D UNet
  - Scalar features processed by fully connected layers
  - 1D features processed by 1D FiLM Convolution layers
  - Initial and final convolutions divided into multiple branches
  - Three head branches for temperature, q000X, and wind vector prediction
  - Classification branch for state_q drops
- **Loss Function:** Huber loss (beta=2)

#### Training
- **Optimizer:** Adam with clip norm
- **Scheduler:** Cosine scheduler for 7 epochs
- **Batch Size:** 384 with lr 0.0012
- **Training Time:** 3 days on RTX 3090 for the entire LowRes dataset

[Kmat pipeline]

### 2nd Stage Modeling
The final submission from our team capaomat (3rd place) is an ensemble of predictions from three members. Some Camaro predictions (ptend_q0001, q0002, q0003) are refined by the 2nd stage. The score improvement is less than 0.0004. Neural network modeling is much more dominant.

#### Features
We used a few features from raw inputs and predictions of 1st stage to prevent overfitting. State_t, state_q, ptend_q, future_state_q features and ratio of future_state_q2 to q3 are provided to the model.

[Fraction of liquid cloud over total cloud as a function of temperature]

#### Model / Training
We employed lightGBM to predict ptend_q at each level. Specifically, we trained the models and updated predictions for various levels. It took less than 20 minutes on CPU to train all 91 models.

### Ensemble
As a final submission, we blended the following 9 models. Our solution achieved the following results:

| exp                                      | Public LB | Private LB | Weight1 | Weight2 |
| ---------------------------------------- | --------- | ---------- | ------- | ------- |
| Camaro1_v2                               | 0.78504   | 0.78199    | 3.0     | 3.0     |
| Camaro2_v2                               | 0.78451   | 0.78159    | 3.0     | 3.0     |
| Camaro3_v2                               | 0.78572   | 0.78154    | 3.0     | 3.5     |
| Camaro4_v2                               | 0.78427   | 0.78141    | 3.0     | 2.0     |
| Camaro5_v2                               | 0.78255   | 0.77957    | 3.0     | 1.0     |
| Pao1                                     | 0.78139   | 0.77770    | 1.0     | 0.5     |
| Pao2                                     | 0.78252   | 0.77864    | 3.0     | 2.0     |
| Pao3                                     | 0.77985   | 0.77801    | 1.0     | 0.5     |
| Kmat1                                    | 0.78120   | 0.77647    | 3.0     | 1.5     |
| Ensemble with weight1 (private best)     | 0.79026   | 0.78810    |         |         |
| Ensemble with weight2 (final submission) | 0.79048   | 0.78792    |         |         |
