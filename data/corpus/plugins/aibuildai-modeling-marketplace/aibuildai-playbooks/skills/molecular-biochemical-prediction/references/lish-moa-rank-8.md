# 8th place solution | From multilabel to multiclass

Competition: lish-moa
Rank: #8
Source: https://www.kaggle.com/c/lish-moa/discussion/200992

A big thank you to the Laboratory for Innovation Science at Harvard and to the Kaggle team for hosting this competition, as well as to all other participants who created such a friendly and yet challenging competitive environment! This was my first DS competition and I learned a lot from you.

### From multilabel to multiclass

The main idea behind my solution was reframing the multilabel prediction problem as multiclass classification to leverage correlations between targets. Even though there were originally 206 binary targets, the training dataset contained only 328 distinct combinations of them. I gave each a class label from 0 to 327, using it as target variable instead. The correspondence between class labels and original targets was stored in a 328x206 matrix.

Then I set up NN models with a softmax activation in the last layer, yielding probability matrices of shape (batch_size x 328) and optimizing for categorical cross-entropy. To retrieve predictions for the 206 original targets, I multiplied the NN output by the 328x206 matrix.

While in theory this "dual" approach is more limited due to the fact that it is impossible to give perfect predictions for a new drug with MoA combinations not previously seen among the 328 classes, in practice these individual models still had decent performance by themselves (best public/private LB scores were 0.01849/0.1633 with NNs and 0.01830/0.01623 with Tabnet).

However, their biggest power came when blending with other models, because this reframing significantly lowered the prediction correlations, as it detects different signals from the training data.

### Final solution

My final solution was a simple average (no weighting) of four models: two Tabnets and two NNs (one multiclass and one multilabel for each architecture). Nothing remarkable about the feature engineering pipeline: I used QuantileTransformer, PCA and row-based statistics as it was done in several public notebooks.

The details of my top selected submission can be seen below (0.01811/0.01603 LB), but I achieved similar private scores of 0.01603 and 0.01604 with other model combinations.

| **Model** |  **Nb folds** | **CV (with ctrl)** | **Public LB** | **Private LB** |
| --- | --- | --- | --- | --- |
| Multiclass NN | 7 | 0.01361 | 0.01884 | 0.01659 |
| Multiclass Tabnet | 7 | 0.01484 | 0.01830 | 0.01623 |
| Multilabel NN | 7 | 0.01453 | 0.01837 | 0.01626 |
| Multilabel Tabnet | 7 | 0.01480 | 0.01834 | 0.01620 |

For instance, one of my blends with four different models (Multiclass NN - 0.01633, Multiclass Tabnet - 0.01636, Multilabel Resnet - 0.01636, Multilabel Tabnet 0.01624) also achieves 0.01603 Private LB, but I did not select it because its public LB score was worse (0.01819).

As CV scores (MultilabelStratifiedKFold) were very different between my models, it was difficult to come up with a final validation strategy, since optimizing blending weights would give 100% to Model 1. In the end, I went with my gut feeling that I should not be overfitting too much as long as things were kept simple, and my two final submissions combined distinct models to minimize risk.
