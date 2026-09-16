# 27th place - Trusting CV always works.

Competition: lish-moa
Rank: #27
Source: https://www.kaggle.com/c/lish-moa/discussion/200630

#### Feature Used
1. <br/>
    - Important features selected from xgboost and RFECV without any preprocessing.
2. <br/>
    -  Variance Threshold on original features. (before scaling and pca)
    - Quantile transformation
    - PCA with 80 gene and 10 cell features
3. <br/>
    - Quantile Transformation
    - PCA with approx 600 gene and approx 60 cell
   - Variance Threshold
4. <br/>
    - Quantile Transformation
    -  PCA with approx 600 gene and approx 60 cell
    -  Variance Threshold
    -  Dropped high correlated features after all the feature engineering
5. <br/>
    -  4000 ML-SMOTE features
    - Quantile Transformation
    -  PCA with approx 500 gene and approx 50 cell
    -  Variance Threshold

#### Little Data Augmentation
- Used albumentations to add random dropout in the features with 0.5 dropout rate and trained
models separately for this data.

#### Models Used (10 fold MultilabelStratifiedKFold with No seeds)
1. 
       Two Three layer neural networks :
         a) (BatchNorm -> Dropout -> WeightNorm(Dense) -> ReLU) * 3
         b) (BatchNorm -> Dropout -> WeightNorm(Dense) -> LeakyReLU) * 3

2. 
      Two headed neural network with skip connections trained with non-scored for transfer learning:
         Head1 -->  out1 = (BatchNorm -> Dropout -> Dense -> LeakyReLU) * 4
         Head2 -->  out2 = out1 +  (BatchNorm -> Dropout -> Dense -> LeakyReLU) * 4
         avg -->  Average([out1, out2])
         final head --> avg + (BatchNorm -> Dropout -> Reshape -> LSTM) * 1
                                         (BatchNorm -> Dense -> LeakyReLU) * 3

3. 
     Two headed neural network with skip connections trained with non-scored for transfer learning:
         Head1 -->  out1 = (BatchNorm -> Dropout -> Dense -> ReLU) * 4
         Head2 -->  out2 = out1 +  (BatchNorm -> Dropout -> Dense -> ReLU) * 4
         avg -->  Average([out1, out2])
         final head --> avg + (BatchNorm -> Dropout -> Dense) * 1
                                         (BatchNorm -> Dense -> ReLU) * 3
4. TabNet.

**Schedulers:** ReduceLROnPlateau, OneCycleLR
**Optimizers:** Adam, AdamW
**Activations:** ReLU, LeakyReLU

(All the models are trained with different above mentioned features and parameters making a total of 16 models.). Submitting average of all these models achieved :
    **CV:0.01501 <br/> Public: 0.01824 <br/> Private: 0.01611**
 
#### Stacking
Stacked predictions from ensemble of all the models by taking simple mean of all the predictions.
Training models with original+stacked features achieved : 
 **CV:0.01495 <br/> Public: 0.01821  <br/> Private: 0.01610**
 

#### Ensemble
Simple average of all models trained with different features (16 models) + models trained with
stacked features (10 models) making an ensemble of total 26 models achieved      
**CV:0.01475 <br/> Public: 0.01819  <br/> Private: 0.01608**
   
#### Best submission (0.01604)
Submission with little blending of weights
[best]

**Thanks to kaggle and all the contributors for sharing their amazing work with notebooks and discussions. I learned so many new things from feature engineering techniques to building models for tabular data and so many more things. Really a great competition so far.**
