# 10th place - NODE (Neural Oblivious Decision Ensembles)

Competition: playground-series-s5e8
Rank: #10
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/10th-place-node-neural-oblivious-decision-ensemble

# 📌 **English Version**

Hi Kaggle community,
I’d like to share a short summary of my final solution that got me into the Top 10 of this competition. 🚀

**General Approach - **

The main idea was to maximize diversity in the first level and use a robust meta-model in the second level to fully exploit the different prediction patterns.

**First Level (Base Models)** 


Xgb - 10 models
Lgbm - 5 Models
CatBoost - 2 Models
Deeptables xDeepFM, DeepFM, AFM
MLP+Bagging
ADAboost+Ext
Logistic Rapids
Xgb+Bagging
KNN Rapids
RF Rapids
Autogluon

All models were trained with KFold validation to generate OOF predictions.

**Second Level (Meta Model)**

The highlight was NODE (Neural Oblivious Decision Ensembles) as the meta-model.
It outperformed alternatives such as Hillclimb, Ridge, Lasso, Optuna-weighted ensembles, and even weighted second-level combinations.


# 📌 **Versão em Português - Aos meus Conterrâneos**

Gostaria de compartilhar um breve resumo da minha solução final que me colocou no Top 10 desta competição. 🚀

**Abordagem Geral**

A ideia principal foi maximizar a diversidade no primeiro nível e usar um meta-modelo robusto no segundo nível para explorar ao máximo os diferentes padrões de previsão.

**Engenharia de Features**

Para variáveis categóricas, usei target encoding e count encoding. Isso permitiu que modelos lineares e redes neurais capturassem informações valiosas sem causar uma explosão de dimensionalidade.

Também experimentei manter as features categóricas em seu formato original para modelos que as tratam nativamente, como CatBoost e LightGBM.

Normalização e padronização (standardization) foram aplicadas apenas em modelos sensíveis à escala das features (ex: MLPs, KNN e regressões lineares).

**Primeiro Nível (Modelos Base)**
XGBoost - 10 modelos
LightGBM - 5 modelos
CatBoost - 2 modelos
DeepTables (xDeepFM, DeepFM, AFM)
MLP+Bagging
AdaBoost+ExtraTrees
Regressão Logística (RAPIDS)
XGBoost+Bagging
KNN (RAPIDS)
Random Forest (RAPIDS)
AutoGluon

Todos os modelos foram treinados com validação K-Fold para gerar previsões OOF (Out-of-Fold).

**Segundo Nível (Meta-modelo)**

O destaque foi o NODE (Neural Oblivious Decision Ensembles) como meta-modelo. Ele superou alternativas como Hillclimb, Ridge, Lasso, ensembles ponderados com Optuna e até mesmo combinações ponderadas de segundo nível.
