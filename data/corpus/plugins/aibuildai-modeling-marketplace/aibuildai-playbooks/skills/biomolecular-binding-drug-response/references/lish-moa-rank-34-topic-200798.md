# Public 46th / Private 34th Solution

Competition: lish-moa
Rank: #34
Source: https://www.kaggle.com/c/lish-moa/discussion/200798

Hello, Kaggler! 
This is Public 46th / Private 34th Solution. 

Thanks to my wonderful teammates, @ttahara @takanobu0210 @masatakashiwagi @taromasuda.
I enjoyed discussion with teammates every weekend!

### Final Submission Notebook
https://www.kaggle.com/ttahara/34th-stacking-5-models-by-mlp-1d-cnn-wo

### Our Pipeline


### What worked
- Add stat feature
- Add PCA feature
- Rankgauss
- Label Smoothing
- Transfer Learning by nonscored for NN, ResNet
- Shallow model
  - Short epoch, learning until limit before loss is NaN by NN
  - n_steps=1, n_shared=1 by TabNet
- Thresholdlng NN
@ttahara customized NN, which preprocesses inputs by element-wise linear functions followed by tanh and feeds the preprocessed values into an ordinary MLP. He calls this model **Thr**esholding NN. 
- Ensemble. In particular, Tabnet and NN's ensemble is effective.
- 2 Stage Stacking by MLP, 1D-CNN, Weight Optimization

### What did NOT work
- SelfSupervised TabNet
- Transfer Learing by nonscored for TabNet
- Feature Selection by TabNet's Feature Importance
- Pseudo Labeling
- Class Balanced loss, focal loss
- PostPredict for large logloss target
- Predict drug id, and label power set.
MoA is determined by drug id, So we tried to predict drug id directly(And target converted to label power set), but not work.
- Using Non-scored as feature
We predict Non-scored MoA, and add Non-scored oof and pred to the train, test data, but not work.

### Not enough time
- Target Encoding to g-,c- bin's feature
- XGBoost, CatBoost, CNN model for single model (Stage 1)
- GCN model for stacking model (Stage 2)
- Netflix Blending
- PostPredict by LGBM
We noticed that there are columns that NN can't predict, but LGBM can (e.g. cyclooxygenase_inhibitor). Therefore, we came up with the idea of repredicting only the columns that are good at LGBM. But not enough time.
