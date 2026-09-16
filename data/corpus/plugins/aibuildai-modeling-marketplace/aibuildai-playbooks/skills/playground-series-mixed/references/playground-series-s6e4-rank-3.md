# Error Diversity Matters: 200-model stacking solution

Competition: playground-series-s6e4
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s6e4/writeups/error-diversity-matters-200-model-stacking-soluti

[Private ascending]
So close, yet so far 😭
[Public ascending]

Hello everyone!

I was very happy to take part in this competition. For me, it turned out to be one of the most interesting and intense Kaggle competitions. Over the course of about a month, I did a huge amount of work to improve my solution and climb as high as possible on the leaderboard. I tried a large number of different approaches, trained many models, created different feature sets, tested various stacking strategies, and spent a lot of time analyzing OOF predictions, model errors, and differences between submission files. In the end, I can say that it was worth it.

In this writeup, I will try to describe the most important parts of my solution. There will be quite a lot of details, because the final pipeline became fairly large and included many experiments. I will try to focus on the ideas that contributed the most and helped gradually improve the result.

Thank you to everyone who reads this writeup to the end. I hope some parts of it will be useful or interesting both for new participants and for Kaggle veterans.

| Model family         | Number of models |
| -------------------- | ---------------: |
| XGBoost              |               28 |
| LightGBM             |               16 |
| CatBoost             |               19 |
| PyTabKit / RealMLP   |                7 |
| TabM                 |               49 |
| H2OAutoML            |                1 |
| KNN                  |                1 |
| ExtraTrees           |                1 |
| Trompt               |                5 |
| HistGradientBoosting |               10 |
| RandomForest         |               10 |
| SVM / SVC            |               15 |
| Logistic Regression  |               21 |
| YDF                  |               10 |
| GraphSAGE GNN        |               10 |
| **Total**            |          **203** |

A big respect to all the people at "@***" who publicly posted the results of their decisions:

* "@cdeotte" for providing the original data exact formula
* 30 "@wguesdon" predictions (xgb, lgb, cat, RealMLP, KerasTab and other models)
* 3 "@kashifalikhan360" (xgb)
* 1 "@ravi20076" (xgb)
* 1 "@utaazu" (catboost)
* 3 "@yunsuxiaozi" (xgb, lgb)
* 1 "@MariaNadeem" (lgb, very good quality work)
* 1 "@jayhawk1900" for catboost solo model
* 10 "@pilkwang" for XGB, LGB, CAT predictions
* 3 "@abdullahsafwan333" for 3 models (xgb, cb and ensemble)

# General ensemble strategy

For the final solution in this competition, I did not use a single standalone model. Instead, the main focus was on an ensemble of a large number of heterogeneous models trained on different feature sets and with different hyperparameters. The main idea was that one strong model is rarely able to cover all types of errors. Therefore, I tried to build a pool of models that not only had good CV, but also produced **different OOF predictions** and made mistakes on different objects. This diversity turned out to be the most useful part of the final ensemble. The overall model pool included different families:

```text
- XGBoost
- CatBoost
- LightGBM
- PyTabKit / RealMLP
- Trompt
- KerasTab / Keras MLP
- HistGradientBoosting
- RandomForest
- Logistic Regression
- SVM / SVC
```

Each model saved OOF predictions for the train set and test predictions for the test set. These predictions were then used as features for the final ensembling. For each base model, I used three class probabilities:

```text
model_c0
model_c1
model_c2
```

For the final combination of OOF/test predictions, I tested three main approaches.

## 1. Single meta-model

The first and main approach was to train one meta-model on the OOF predictions of the base models. If the ensemble contained `N` base models, each of them produced 3 probability features, and the final matrix for the meta-model had the size:

```text
N × 3 features
```

As meta-models, I tested:

```text
- LightGBM
- CatBoost
- XGBoost
```

All of them were trained on the OOF predictions of the base models and then applied to the test predictions. The best result among these options was achieved by the **LightGBM stacker**. It turned out to be the most stable and used the base model probabilities better than the other meta-models. Therefore, the main final single-stacker was built with LightGBM. This option was the strongest among all tested ensemble approaches.

## 2. Blending two meta-models: Boosting + Ridge/Lasso

The second approach was to use two different meta-models and then blend their predictions. The idea was to combine a nonlinear boosting-based meta-model with a regularized linear meta-model. As boosting meta-models, I used LightGBM, CatBoost, and XGBoost. As a regularized linear model, I tested options close in spirit to Ridge, Lasso, and Logistic Regression with L1/L2 regularization. The boosting model could capture nonlinear dependencies between base model predictions, for example situations where a certain model should be trusted only under a specific combination of probabilities from other models. The linear model, on the other hand, acted as a more stable and regularized way to assign weights to base model predictions. It is less prone to overfitting and works well with strongly correlated meta-features, which OOF predictions from different models usually are. After training the two meta-models, their predictions were blended:

```text
final_prediction = alpha * boosting_prediction + (1 - alpha) * linear_prediction
```

This approach was useful as an additional robustness check for the ensemble, but the best result still came from the single LightGBM meta-model.

## 3. OOF prediction selection and weighting via "Hill Climbing"

The third approach was hill climbing ensemble selection. In this option, I did not train a full meta-model, but instead tried to select a subset of models and weights for their predictions directly using OOF. The general idea of the algorithm was:

```text
1. Start with the best single model or a simple ensemble;
2. At each step, try adding one of the remaining models;
3. For each model, tune a small weight;
4. Keep only the addition that improves the OOF score;
5. Repeat the process as long as the score improves;
```

This approach makes it possible to automatically select models that truly help the ensemble and ignore models that hurt the result or fully duplicate already existing predictions. Hill climbing was useful for analyzing which models provided an additional signal. However, in the final result, it was worse than the LightGBM stacker in both LB and CV score. The main reason is that LightGBM could use more complex nonlinear relationships between model predictions, while hill climbing mainly selected weights for mixing probabilities. The "hill climbing" algorithm is also more prone to overfitting than a single LightGBM model.

For this competition, the best approach was not simple averaging and not hill climbing, but specifically a **LightGBM stacker** trained on a large and diverse pool of OOF predictions. The final success was related not only to the quality of individual models, but also to the fact that different models made different errors. The LightGBM meta-model was able to effectively use this diversity and select the most useful signals from a large number of base predictions.

# Electrical Resistivity of the medium (secret feature)

I built the first boosting models myself, and at this stage one of the most useful features turned out to be a feature related to the **electrical resistivity of the medium**, calculated from electrical conductivity. Since my background and experience are related to geophysics, I tried to interpret some of the features not only as ordinary tabular variables, but also from the point of view of the physical properties of the medium. In particular, for soil, an important characteristic is the relationship between electrical conductivity and electrical resistivity. Electrical conductivity and electrical resistivity are reciprocal quantities:

```text
ρ = 1 / σ
```

where:

```text
ρ - electrical resistivity of the medium;
σ - electrical conductivity of the medium;
```

If electrical conductivity is given in `S/m`, then resistivity is obtained in `Ω·m`:

```text
ρ [Ω·m] = 1 / σ [S/m]
```

This feature turned out to be useful because it transformed the original electrical conductivity into a more physically interpretable quantity. For tasks related to soil and irrigation needs, this makes sense: the electrical properties of soil depend on moisture, mineralization, the composition of pore fluid, soil type, clay content, and other factors. In geophysics, it is important to distinguish between **true electrical resistivity of the medium** and **apparent resistivity**. **True resistivity** is a physical property of a specific homogeneous medium or layer. It characterizes the ability of a material to resist the flow of electric current and is defined as the reciprocal of electrical conductivity. In the ideal case, if the medium is homogeneous and isotropic, this value directly describes the electrical properties of that medium. **Apparent resistivity** is a quantity obtained from field electrical prospecting measurements. It depends not only on the true resistivities of rocks or soils, but also on the geometry of the measurement array, the heterogeneity of the medium, the depth of investigation, and the distribution of currents in the ground.

In the simplest form, apparent resistivity is written as:

```text
ρa = K × ΔU / I
```

where:

```text
ρa - apparent resistivity;
K - geometric factor of the array;
ΔU - measured potential difference;
I - current;
```

For different electrical prospecting arrays, the coefficient `K` is calculated differently. For example, for the Wenner array:

```text
K = 2πa
```

and then:

```text
ρa = 2πa × ΔU / I
```

where `a` is the distance between neighboring electrodes.

The main difference is that:

```text
true ρ is a property of the medium
apparent ρa is a measurement result that depends on the medium and the acquisition geometry
```

If the medium is homogeneous, apparent resistivity may be close to true resistivity. But if the medium is heterogeneous, for example if the soil has different horizons, different moisture levels, or different mineralization with depth, then `ρa` is an integral response of the medium. For this task, the feature related to resistivity turned out to be useful in the first boosting models. It allowed me to represent electrical conductivity in a form that better reflects the physical state of the soil. In other words, such a feature could indirectly help the model better separate states related to irrigation needs. In tabular models, this showed up practically: transforming electrical conductivity into resistivity became one of the strong engineered features and helped improve the early boosting models. This feature was especially useful at the beginning of the competition, when the main ensemble was still built around XGBoost, CatBoost, and LightGBM. This example clearly shows that even in a machine learning competition, domain knowledge can provide a strong improvement. In my case, the geophysical interpretation of the electrical conductivity feature helped create one of the most useful physical features for the first models.

# Additional Model Families and Feature Diversity

The initial ensemble was mainly based on strong gradient boosting models and neural tabular models:

```text
XGBoost + CatBoost + LightGBM + RealMLP(PyTabKit)
Public LB ≈ 0.98110
```

After that, the main goal was not simply to add models with higher CV, but to increase the diversity of OOF prediction errors. The final stacker was trained on OOF predictions from a large number of different model families, and the main conclusion was that models with different error patterns were much more useful than models with slightly higher CV. The final meta-model was a LightGBM stacker trained on the probabilities of the base models. Each base model added three probability columns, one for each class. All predictions were converted to a unified class order before stacking. **I also concluded that with a high CV, for example 0.98172, I got an LB of about 0.98126, while with CV = 0.98110 I got an LB close to 0.98200. From this, I concluded that the 20% of test data included in the public submission is too noisy or heterogeneous. In any case, this conclusion can be disproved after analyzing the private leaderboard.**

# Trompt and KerasTab Models

After the initial boosting models and PyTabKit, I added Trompt and KerasTab models:

```text
XGB + CatBoost + LightGBM + MLP(PyTabKit)
Public LB ≈ 0.98110

+ Trompt and KerasTab
Public LB ≈ 0.98121
```

The goal of these models was to add neural tabular predictions that differ from tree-based models. Even if their CV score was not always higher than that of boosting models, their OOF errors were different enough to improve the prediction of the final meta-model.

## Trompt

Trompt was used as a neural tabular model based on the public notebook:

```text
https://www.kaggle.com/code/yekenot/ps-s6-e4-trompt-pytorch-frame
```

The original notebook was used as a base, but I modified the feature generation and trained several variations. For Trompt, I used several feature sets. The goal was not to create one best Trompt model, but to obtain several models with different inductive biases and different error patterns.

The feature sets included different combinations of:

```text
- original numerical features;
- original categorical features;
- discretized numerical features;
- cyclic encoding for selected numerical variables;
- simple interaction features;
- transformed numerical features;
- model-specific feature subsets;
```

## KerasTab / Keras MLP

I also added Keras-based tabular neural networks. These models used a mixed architecture for numerical and categorical features:

```text
numerical branch:
    numerical features
    -> BatchNormalization
    -> Dense layers

categorical branch:
    categorical features
    -> Embedding layers
    -> concatenation with numerical branch

final trunk:
    Dense layers
    -> Dropout
    -> Softmax output
```

**I was surprised, but simple fully connected neural networks gave a very good CV score (+-0.97700), on par with boosting models.** One of the strongest Keras models was `keras_mlp_v1`. Its predictions later received very high importance in the LightGBM stacker.

The `keras_mlp_v1` feature set was based on:

```text
1. All original numerical features
2. All original categorical object features
3. Cyclic encoding for Rainfall_mm:
   - sin(2π * Rainfall_mm / 100)
4. Cyclic encoding for Soil_Moisture:
   - sin(2π * Soil_Moisture / 4)
5. Floor-based categorical versions of all original numerical features
6. Quantile bins for Temperature_C:
   - 5 quantile bins
   - 40 quantile bins
```

The model used embeddings for categorical features and dense layers for numerical features. Keras models were useful because they learned dependencies different from gradient boosting models. In particular, embedding-based processing of categorical features and the dense numerical branch created a different type of decision boundary.

# HistGradientBoosting and RandomForest

After adding Trompt and KerasTab, I added HistGradientBoosting and RandomForest:

```text
XGB + CatBoost + LightGBM + MLP(PyTabKit)
+ Trompt and KerasTab
Public LB ≈ 0.98121

+ HistGradientBoosting and RandomForest
Public LB ≈ 0.98145
```

This gave a noticeable improvement. The reason is that these models added another type of tree-based signal, different from XGBoost, CatBoost, and LightGBM.

## HistGradientBoosting

HistGradientBoosting was trained on several feature sets. Some of them were similar to the feature sets for Trompt/KerasTab, while others were created specifically for HistGradientBoosting.

The feature sets included:

```text
- original numerical features;
- encoded categorical features;
- discretized numerical features;
- bin-based features;
- cyclic encodings;
- interaction features;
```

HistGradientBoosting was useful because it behaved differently from the main boosting models. Although it is also a tree-based method, histogram-based splitting and the sklearn implementation produced different prediction patterns. Some HistGradientBoosting models were not the best individual models, but they helped the stacker because their errors did not match the errors of XGB/CatBoost/LightGBM.

## RandomForest

I also added RandomForest models, including GPU-based cuML RandomForest and several CPU-trained RandomForest models. The RandomForest models were trained on several feature sets. RandomForest is usually weaker than gradient boosting in tabular competitions, but in this ensemble it was useful because it added completely different errors. RandomForest (a supervised machine learning blending method) often makes different mistakes than boosting models, especially near decision boundaries. The RandomForest models were added not because they were the strongest individually, but because they increased ensemble diversity. The model itself is trained in parallel on many independent trees, and the final prediction is obtained by weighted voting of all trees. This is different from gradient boosting methods, where each model is trained sequentially on the errors of previous models. The feature sets were intentionally made different from the main boosting features and included:

```text
- original features;
- categorical encodings;
- discretized numerical features;
- selected engineered features;
- feature subsets with different levels of noise and interaction features;
```

# Logistic Regression and SVM / SVC

The biggest improvement came after adding linear models:

```text
XGB + CatBoost + LightGBM + MLP(PyTabKit)
+ Trompt and KerasTab
+ HistGradientBoosting and RandomForest
Public LB ≈ 0.98145

+ Logistic Regression and SVM/SVC
Public LB ≈ 0.98201
```

This was the most important stage in the development of the ensemble. At first glance, linear models may seem too simple for this task. But they turned out to be very useful because they have a completely different bias compared to trees and neural networks. Logistic Regression was trained on a large number of different feature sets. The feature sets included:

```text
- original numerical features;
- scaled numerical features;
- categorical encodings;
- discretized numerical variables;
- bin features;
- interaction features;
- feature sets inspired by HistGradientBoosting models;
- additional manually designed feature groups;
```

The Logistic Regression models were trained mainly for diversity. Their standalone performance was not necessarily higher than that of boosting models, but their prediction errors were different. This made them useful for the final LightGBM stacker. For these models, scaling was important. Linear models are sensitive to feature scale, so numerical features were normalized or standardized depending on the feature set.

## SVM / SVC Models

SVM models turned out not to be a very useful addition. SVM models were expensive to train. Even with a relatively small number of features, training one SVM model could take more than two hours. Nevertheless, in combination with previous models, they improved the CV score well, but strongly reduced LB with different feature combinations. Again, going back to the beginning, it is possible that the lower LB is related to the heterogeneity of these 20% of data and that SVM is actually a useful addition. I trained both nonlinear SVM/SVC and linear SVM variants.

The SVM family included:

```text
- RBF-kernel SVM/SVC models;
- linear SVM models;
- multiple feature sets;
- different hyperparameter values;
- GPU-based implementations where possible;
```

# YDF / GraphSAGE GNN

In addition to the models that improved the final ensemble, I also experimented with YDF and GraphSAGE GNN. However, these models were not used in the final solution because, in combination with already strong models, they noticeably worsened the Public LB. Even when added to the strongest ensemble, the result dropped approximately from LB = 0.98201 to LB ≈ 0.98130. Also, in the feature importance table of the final LightGBM stacker, these models had very weak `gain`, which meant that the meta-model almost did not find useful additional signal in their OOF predictions.

YDF, or Yggdrasil Decision Forests, is a library for training decision-tree-based models. It supports different tree-based algorithms, including Random Forest and Gradient Boosted Trees, and is well suited for tabular data. In this task, YDF models were trained as an additional tree-based family to add predictions with behavior different from XGBoost, CatBoost, LightGBM, HistGradientBoosting, and RandomForest. I often noticed the use of this model in previous competitions, but in practice, YDF predictions turned out to be poorly compatible with the already strong ensemble. When YDF was added to the LightGBM stacker, the final LB decreased. According to the analysis of submission files, YDF often shifted predictions toward the `Medium` class, removing some `High` and `Low` predictions that were important for a strong leaderboard score. Because of this, YDF was excluded from the final model set.

From the previous "Churn Prediction" competition, I studied in detail the very cool work of **"@cdeotte"**. Among his approaches, I really liked the GraphSAGE GNN model. GraphSAGE GNN is a graph neural network that learns representations of objects by aggregating features from neighbors in a graph. In this task, I built the graph not as a natural network of objects, but as an artificial similarity graph between rows of tabular data.

The idea of this algorithm was as follows:

* represent train/test rows as graph nodes;
* connect similar objects through a KNN graph;
* train GraphSAGE so that the model uses information from neighbors;
* obtain OOF/test predictions different from ordinary tabular models;

This approach was supposed to add a new inductive bias: instead of processing each row independently, the model could take into account the local structure of the data through neighbors. However, GraphSAGE GNN also did not improve the final ensemble. Although separately its OOF predictions looked quite interesting, in combination with strong models the graph did not provide a useful LB improvement. In the feature importance of the final stacker, the features corresponding to GNN predictions had low `gain`. This meant that the LightGBM meta-model rarely used these features for important splits. Therefore, GraphSAGE GNN also did not enter the final ensemble. **Again, it is possible that this model is very useful for the ensemble, but without knowing the heterogeneity of the test set in the public part, I cannot say anything for sure; I can only rely on the analysis of the standalone GNN models and their importance in the ensemble.**

These experiments were useful because they confirmed an important principle: not every new model family improves the ensemble, even if it theoretically adds diversity. Therefore, in the final solution I kept only those models that either improved LB while maintaining good CV > 0.98110, or provided a stable useful signal in the stacker feature importance.

# Feature Set Philosophy

The key idea of the project was not to create one universal feature set. Instead, each model family received its own feature sets. Some feature sets were used in several models, but many were created specifically for a particular model type.

For example:

```text
- neural models benefited from embeddings, scaled numerical features, and smooth transformations;
- linear models benefited from scaling, discretization, and interaction features;
- tree models benefited from raw features, bins, categorical encodings, and nonlinear splits;
```

As a result, the ensemble used a large number of OOF prediction files from different model families. The final LightGBM stacker did not use the original raw features directly; instead, it used the prediction probabilities from all base models.

# Final Progress

The approximate public leaderboard improvement after adding different model families was as follows:

```text
XGB + CatBoost + LightGBM + MLP(PyTabKit)
LB ≈ 0.98110

+ Trompt and KerasTab
LB ≈ 0.98121

+ HistGradientBoosting and RandomForest
LB ≈ 0.98145

+ Logistic Regression
LB ≈ 0.98201
```

# Final Progress

The approximate public leaderboard improvement after adding different model families was as follows:

```text
XGB + CatBoost + LightGBM + MLP(PyTabKit)
LB ≈ 0.98110

+ Trompt and KerasTab
LB ≈ 0.98121

+ HistGradientBoosting and RandomForest
LB ≈ 0.98145

+ Logistic Regression
LB ≈ 0.98197
```

The final improvement did not come from one single model. It came from combining a large number of different model families whose OOF predictions had complementary errors. The most useful additions were not always the models with the highest standalone CV, but the models that provided new information to the LightGBM stacker. As the main final single-stacker solution, I used all models presented in the initial table except the **SVM/SVC**, **YDF**, and **GraphSAGE GNN** model families. These models were tested separately, but did not enter the final single-stacker because, in combination with the already strong ensemble, they worsened the public leaderboard score or had too weak a gain in the LightGBM stacker feature importance table.

The final result of this ensembling on the selected set of models was:

```text
LB = 0.98197
```

I managed to get an additional improvement to:

```text
LB = 0.98201
```

not by adding new standalone models, but by using final voting between several strong submission files. The voting was built around the best validated solutions. A strong LightGBM stacker submission was used as the base anchor, and then very cautious guarded voting / override rules were applied to it. The idea was to change the anchor prediction only in those rows where several strong submissions confidently agreed against the original class.

I tested different voting variants:

```text
- majority voting between the best submission files;
- weighted voting with weights based on public LB score;
- guarded override, where the class was changed only under strong consensus;
- variants where Medium->High and High->Medium transitions were controlled separately;
```

The best result came from a very conservative guarded voting approach. It did not try to fully replace the predictions of the main stacker, but only corrected a small number of controversial rows where several strong models agreed with each other. This approach turned out to be more useful than adding new models with higher local CV, because it broke the already strong class distribution less and matched the public leaderboard better. The best final result after this voting was:

```text
LB = 0.98201
```

Thus, the main improvement up to `0.98197` was achieved by building a diverse pool of models and a LightGBM meta-model, while the final small improvement to `0.98201` came from careful voting / guarded override on top of the best submission files.

# Computing Resources

I did not have access to powerful external computing resources. All GPU training was done inside Kaggle. The weekly Kaggle GPU limit was enough for me because I trained GPU and CPU models in parallel. Some model families trained relatively quickly, but SVM models were expensive. In particular, some SVM models and some implementations of XGB/LGB/CatBoost were among the most GPU memory-intensive parts of the pipeline. Even with a relatively small number of features, training one SVM model could take more than two hours. Nevertheless, these models were valuable because they increased the diversity of the final ensemble.

**Thank you for your attention!**
