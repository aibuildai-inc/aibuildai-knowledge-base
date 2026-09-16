# 15th Place Solution Meta features ,FE, DART, CAT, XG , Tabnet , MLP , ensemble 😊

Competition: amex-default-prediction
Rank: #15
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/347908

Thanks first to Kaggle for hosting this interesting competition  , I was personally interested in this as it cynosure with the financial domain which I spend majority of my time working on . We had a great team with @sirius81 @liji11 @tonymarkchris and @hanzhou0315  who each brought there unique skills to the competition.. A big thank you to all of them .

# Feature Engineering
Our feature engineering structure was influenced by this great notebook [rapids-cudf-feature-engineering-xgb](https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb) from @jiweiliu 

## Meta Features
Our meta features were the differentiations to boost our ensembles .These are similar and well described here [12th Place Solution](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347786) and a big thanks for @sirius81 to come up with those . As Sirius also mentioned in that post we used them a s 13 numerical features per CID after flattening .

## For models without meta features
- Great features suggested by Ragnar helped like last-mean features and last-min/max features for some columns, also we did apply diff 1,2 lags as well .After pay features were also helpful .
- We did usual aggregates std,mean,min,max for categoricals (nunique,count,first,mean). We also tried MAD (Mean Absolute deviation) and it did help our cv though Public LB was bit less so we didn't include the model based on mad in final scores though it could have helped in hindsight ,
- We also did add percentage change features , which basically do the percentage change on numerical features and thought they would work better than just a numeric diff . They did help us in some models .
- Also one feature that was there in most of our good models was keeping last, first and middle component of the statements as we hypothesized that will help us cover most variation for a customer as we had aggregated features .
- Another feature we tried was to say calculate spend/balance or spend_sum/balance_sum ratios , It did help in some models but not most .
- Trimming of non impactful features also did help to reduce number of features in 1k-2k range .

## Hybrid approach (meta+aggs) : 
We also gave a shot at mixing the flattened meta features with aggregates and it did help our models specially LGBM CAT XG and Tabnet.
Cause of meta features this usually converged faster (fewer rounds with early stopping in XG) so we had to **Lower LR**  for this approach .

#Models
All models were trained on meta features and/or engineered aggregates . NN were scaled for the most part with GaussianScalar

**LGBM (Max cv 0.79932 Private 0.80731)**: This was as with most based on notebook by @ragnar123 . We did do little tweaks to the hyperparams but most were similar .Here as noted in discussions lowering the LR certainly helped in the case of DART and also us adding meta features (around *0.0075 LR*).
**XG (Max CV 0.7984 Priv. 0.80687)**: XG was based on @jiweiliu  great notebook
**CAT (Max CV 0.7972)**: Cat we did tune a bit with below params and it helped us a lot .
```python
CatBoostClassifier( random_state=CFG.seed,  
                                bootstrap_type='Bernoulli',
                                task_type="GPU",
                                devices='0:1',  
                                use_best_model = True, 
                                iterations = 11500,
                                num_leaves =64,
                                subsample = 0.74,
                                grow_policy = 'Lossguide', 
                                depth = 9) 
```
**Tabnet (MAX CV 0.793133 Private 0.80447)**: Tabnet used mostly standard params and did give better results with meta features .
**MLP (Private 0.80109).** We tried models with tabnet using cat embeddings and without (by onehotencoding). Also **GaussianScaler** helped with scaling of numerical features in case of NN and tabnet
**AutoML (MAX CV 0.79714)**: @liji11 tried automl which did help our ensemble 
**Tabformer ( MAX CV 0.79507 Private 0.80480)** : This mostly based on the public [notebook](https://www.kaggle.com/code/gauravbrills/tabtransformer-training)  with more heads as per the paper . This we noticed was better than tabnet in private LB but we didnt choose it as was not working well in ensemble 😑. Hyperparams below 
```python
NUM_TRANSFORMER_BLOCKS = 6  # Number of transformer blocks. 6 paper recommends
NUM_HEADS = 8  # Number of attention heads. 8 Heads paper recommends
EMBEDDING_DIMS = 16#10  # Embedding dimensions of the categorical features. check 16,32
DROPOUT_RATE = 0.1
MLP_HIDDEN_UNITS_FACTORS = [
    4,
    2,
]  # MLP hidden layer units, as factors of the number of inputs. =>(4,2)
NUM_MLP_BLOCKS = 4  # Number of MLP blocks in the baseline model. Paper 4
MLP_ACTIVATION = keras.activations.selu
```

**TCN MLP**: TCN over 13 sequences , Not chosen Max CV 0.78
Goes like this inspired by the amex paper 
```python
   embeddings = []
    for k in range(11): 
      vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[cat_cols[k]]
      #print(f"cat {cat_cols[k]} index {k} len {len(vocabulary)}")
      emb = tf.keras.layers.Embedding(len(vocabulary),EMBEDDING_DIMS)
      embeddings.append(emb(inputs[:,:,k]))
    in_ = tf.keras.layers.Concatenate()([inputs[:,:,11:]]+embeddings)  
    activation = 'swish'
    l1 = 1e-7
    l2 = 4e-4
    reg = 4e-4
    # SIMPLE Wavenet TCN BACKBONE  
    _x = BatchNormalization()(in_)
    x = TCN(nb_filters=256, kernel_size=4,return_sequences=False, dropout_rate=0.0, dilations=[2 ** i for i in range(9)])(_x)
    #x = TCN(nb_filters=128, kernel_size=3,return_sequences=False,use_layer_norm=True,dropout_rate=0.05,dilations=[2 ** i for i in range(7)])(x)
    x0 = Dense(128, 
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
#                activity_regularizer=tf.keras.regularizers.L1L2(l1=l1,l2=l2),
              activation=activation,
             )(x)
    x0 = Dropout(0.1)(x0)
... MORE OF MLP ....
x_output = Dense(1,
              activation='sigmoid',
             )(x)
```

# Ensemble techniques

All ensembles were done on log odds as we found it worked better when ensembling NN models . We had also done ensemble with rank method but that didnt work well when ensembling with NN .

We tried a bunch of meta ensemble techniques with are many models .The ones that worked were [forward selection](https://www.kaggle.com/code/cdeotte/forward-selection-oof-ensemble-0-942-private) by @cdeotte , Optuna weights and Elasticnet on top of oof preds . We also tried stacking and voting techniques but they weren't very successful.

# What did not work or DID ?

- [Tabformer](https://www.kaggle.com/code/gauravbrills/tabtransformer-training) : We finally made tabformer reach 0.795 cv but it was not part of our final submission. This I will say *did work* but did not ensemble well though later we realized was our best Private LB scoring NN .
- TCN wavenet : We tried to use 13 customer history sequences in a tcn wavenet + MLP implementation but results were only around 0.78 ish so we dropped the idea .
- Saint and widedeep models : We also gave a stab on wide deep package to try out  [SAINT](https://arxiv.org/abs/2106.01342)  but were not able to get far with that . reference [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep)
- Some features as described above and some dropped based on permutation or zero feature importance .
- We tried to do multi seed ensembles for all our good models . This somehow did not give us good results at the end compared to sticking with 42 seed.
- Sirius also tried Target encoding at the end but we had really very little time to check on these in ensemble .
- We also tried GAN techniques to impute missing values but dropped them at the end .

Finally luckily we selected a good enough final submission for 🏅(There was as usual a lot of confusion thanks @tonymarkchris for voting for this 😄), though seems some with lower LB were better so should have trusted cv a bit more :) 

**✅  Best selected submission Private LB 0.80838 ( Optuna weighted models , xg, tabnet.mlp,lgbm,cat and automl)**
**📮 Best submission : Private LB 0.80852 ( Optuna xg,cat,tabnet,mlp,automl and lgbm)**

Thanks for reading 😄
