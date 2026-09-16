# 43rd Place : Summary and What Worked Well

Competition: open-problems-multimodal
Rank: #43
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/368421

Many thanks to the organizers for this interesting multimodal challenge and giving us unique multi-target time series dataset to ideate upon.

### **Data Preparation and Feature Pipeline**

- TruncatedSVD of both cite and multi inputs - 100 components
- Binarized data TruncatedSVD of both cite and multi inputs - 100 components
- TruncatedSVD of multi targets - 256 components
- PCA of both cite and multi inputs - 40 components - used only in some of the models for additional features
- Most correlated raw features for individual cite targets
- Usage of '*Day*' as a feature

### **CV Scheme**

- GroupKFold by donor for both cite and multi - used with higher weightage in the final pipeline
- KFold for both cite and multi - since it was also correlated, kept it in the final pipeline with low weightage

### **Modeling Pipeline**

- MLPs with varied number of layers without binary components for both cite and multi (0.813 on public)
- MLPs with a mixture of both binary and non binary components for both cite and multi (0.813 on public)
- TabNet and LGBM model with dimensionality reduced cite data and multi (0.812 on public)
- Individual models with *highly correlated important features* per target with LGBM, XGB and CB for cite data (jump to 0.8142 on public, resulted in best ensemble)

#### **TakeAways**

- pyBoost
- day similarity analysis (as @l0glikelihood trained only on day 7 for multiome)

Would have expected to go upward with the shakeup, realized that other teams really did great and many congratulations to them. It was a wonderful competition, one of it's kind. 

Cheers!
