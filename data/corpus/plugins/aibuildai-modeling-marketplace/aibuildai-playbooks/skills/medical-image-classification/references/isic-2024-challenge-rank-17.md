# Tree + NN with PB 176 LB 185

Competition: isic-2024-challenge
Rank: #17
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/533005

https://www.kaggle.com/code/goldenlock/tree-withnn-multimodal-pb176?scriptVersionId=195993365  
I might release simplifed version with full code later, currently you could find code from isic-2024-code dataset.    
Notice late sub results does not mean anything else as I will overfit PB to choose models, but still it could help us learn.
1. LGB with group feats   
   - Group by patient 
   - Group by patient + loc (loc, simple loc, general site)
   - Rank feature including zscore, 0-1 minmax score, group rank score
   - Remove attribution and using region instead(Unknown hospitals in AUS)
    LGB with meta only could get PB 165, LB 177, CV 1751  (single model lgb only, 10 folds)
https://www.kaggle.com/code/goldenlock/tree-metaonly?scriptVersionId=195834418  
2. NN model with image only
    Using efficientnet_b1, with multi softmax objects of target,iddx_2,iddx_3,iddx_4. And alsol split neg using iddx_2.  
NN model alone could get PB 153, LB 161, CV 15843  (single model, 10 folds)
https://www.kaggle.com/code/goldenlock/imgonly?scriptVersionId=195850404  
3. LGB with nn model pred as feature   
https://www.kaggle.com/code/goldenlock/tree-withnn?scriptVersionId=195864320
PB 173, LB 185, CV 1816  (single model 10 folds, image model is also single model 10 folds)
4. NN model with meta data using dot pooling 
Same features as LGB model but using nn end2end train.
https://www.kaggle.com/competitions/isic-2024-challenge/discussion/532570  
https://www.kaggle.com/code/goldenlock/multimodal-pb172?scriptVersionId=195819415  
PB 172, LB 178, CV 17889 (single model 10 folds)
5. Ensemble of LGB and NN
with weight 1 both 
https://www.kaggle.com/code/goldenlock/tree-withnn-multimodal?scriptVersionId=195885581  
PB 175, LB 185, CV 18239
6. More Ensembles
  - Image Only models: effnet_b1 and caformer_s18
  - Image meta models: 2 dot pooling models, with one using bin embedding for each non cat features which has higher cv 180
PB 176, LB 185
https://www.kaggle.com/code/goldenlock/tree-withnn-multimodal-pb176?scriptVersionId=195993365
