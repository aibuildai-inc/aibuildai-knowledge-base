# 1st Place of Jane Street 🏆 ➜ Adapted to Crypto 👌

Competition: g-research-crypto-forecasting
Rank: #1
Source: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/286676

# <span class="title-section w3-xxlarge" id="codebook">1st Place of Jane Street 🏆 ➜ Adapted to Crypto 👌</span>

> **TL;DR:** [Here it is](https://www.kaggle.com/yamqwe/1st-place-of-jane-street-adapted-to-crypto)

Hi Guys, 

I just finished converting the [1st place winning solution](https://www.kaggle.com/gogo827jz/jane-street-supervised-autoencoder-mlp) of Yirun Zhang from [Jane Street](https://www.kaggle.com/c/jane-street-market-prediction) market prediction competition earlier this year. 

The notebook is [right here](https://www.kaggle.com/yamqwe/1st-place-of-jane-street-adapted-to-crypto).

It is a simple starter notebook using purged group time series. There are many configuration variables to allow you to experiment. Use either GPU or TPU. You can control which years are loaded, which neural networks are used, and whether to use feature engineering. 

You can experiment with different data preprocessing, model architecture, loss, optimizers, and learning rate schedules. The extra datasets contain the full history of the assets at the same format of the competition, so you can input that into your model too.

____ 

**The solution: Encoder + MLP**:



The main idea of using an encoder is the denoise the data. After many attempts at using an unsupervised autoencoder, the choice landed on a bottleneck encoder as this will preserve the intra-feature relations.

**Main Ideas:**

In Jane street competition, Yirun has used a supervised autoencoder MLP approach and his teammates use XGBoost. Their final submission is a simple blend of these two models. Here, We explain Yirun's approach in detail.

The supervised autoencoder approach was initially proposed in [Bottleneck encoder + MLP + Keras Tuner](https://www.kaggle.com/aimind/bottleneck-encoder-mlp-keras-tuner-8601c5) and was adapted to this competition [here](https://www.kaggle.com/yamqwe/bottleneck-encoder-mlp-keras-tuner), where one supervised autoencoder is trained separately before cross-validation (CV) split. Yirun have realized that this training may cause label leakage because the autoencoder has seen part of the data in the validation set in each CV split and it can generate label-leakage features to overfit. So, his approach was to train the supervised autoencoder along with MLP in one model in each CV split. The training processes and explanations are given in the notebook and the following statements.

**The Model:**

- Use autoencoder to create new features, concatenating with the original features as the input to the downstream MLP model
- Train autoencoder and MLP together in each CV split to prevent data leakage
- Add target information to autoencoder (supervised learning) to force it to generate more relevant features, and to create a shortcut for backpropagation of gradient
- Add Gaussian noise layer before encoder for data augmentation and to prevent overfitting
- Use swish activation function instead of ReLU to prevent ‘dead neuron’ and smooth the gradient
- Batch Normalisation and Dropout are used for MLP
- Use Keras Tuner to find the optimal hyperparameter set

This notebook follows the ideas presented in my "Initial Thoughts" [here][1]. 

Happy Kaggling! 
Cheers ^^
