# 3rd Place Solution (+ github code)

Competition: trends-assessment-prediction
Rank: #3
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162934

First of all,  I want to thank hosts for organizing this interesting competition. This competition was a lot of fun because we got to try many techniques! And I'm so happy with my first solo gold!

# Solution overview
.png?generation=1593524751624771&amp;alt=media)

# Solution Details

I think that the point of this competition is stacking and diversity of models.
So, I built many models as shown below.

### 1. Simple models by table feature.
First, I trained simple models by fnc and loding feature.
- SVM (rbf kernel and linear kernel)
- NuSVM
- KNN
- Ridge
- BaggingRegressor(Ridge)

### 2. NN models by fMRI data.

### 2-1. 3D voxel

I applied sample wise and component wise normalization.

- ResNet18 replaced by 3D modules.
- Simple 3dConvNet (smaller parameter than ResNet18)
  - trained by all label
  - trained by age, domain1_var1 and domain1_var2
- (2 + 1)D CNN (suggested in [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf))
  - better than ResNet18

### 2-2. masked data
[Masker of Nilearn](https://nilearn.github.io/modules/generated/nilearn.input_data.NiftiLabelsMasker.html#nilearn.input_data.NiftiLabelsMasker) can extract signals from brain parcellation. Then, I used [schaefer_2018](https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_atlas_schaefer_2018.html#nilearn.datasets.fetch_atlas_schaefer_2018) parcellation and extracted 400 rois from fMRI. (reduce shape [53, 52, 64, 53] to [53, 400])
Finally, I built some models by this [53, 400] signals.

- CNN models (input channel is 400)
 - 1D ResNet18
 - 1D densenet121
 - 1D ResNest 14d
- Transformer
  - No PositionalEncoding
  - Self Attention of components by components (feature axis is 400)
- GIN (Graph Isomorphism Network)
  - Impremented by [Deep Graph Library](https://www.dgl.ai/)
  - I used the network architecture like in [this paper] (https://arxiv.org/pdf/2001.03690.pdf).

### 3. LGBM and XGBoost with table features, NN features and voxel statistical features.
- I computed voxel statistical feature (mean, max, kurt, skew) of each components.
  - I used NiftiMasker for extracting non-zero area.
- LGBM(XGBoost) model with hidden feature of (2 + 1)D CNN
- LGBM(XGBoost) model with hidden feature of Simple 3dConvNet
- LGBM(XGBoost) model with hidden feature of GIN

I selected 1024 feature of each models by feature importance and reduced the number of leaves to avoid over-fitting (n_leaves = 2).

### 4. Stacking

I just looked at the distribution of predictions in the training and test data and erased them if there were too much discrepancy. And I made interaction feature between domains (sum, abs(diff), multiply). After the preprocessing, I used Linear SVM and LGBM for stacking and finally blending the predictions.

# Some Insights

- 3DCNN with table feature had better CV but this model was useless when stacking.
- Removing noise prediction had important role on shake-up.
- MONAI framework is too slow. [Rising](https://github.com/PhoenixDL/rising) is faster and I used it. But I don't know if it helped to boost the score...

# What didn't work

- Pseudo label
- Ridge stacking, XGBoost stacking, Second level stacking
- GNN with fnc attention
- Select less feature of lgbm and xgboost than 1024

[Updated]
I uploaded code to https://github.com/shimacos37/kaggle-trends-3rd-place-solution with Dockerfile.
This code uses several tools. 
1. I used [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) to lap training and evaluation code and [hydra](https://github.com/facebookresearch/hydra) to manage parameter.
1. I used [Weights &amp; Biases](https://www.wandb.com/) to manage expetiments and GCS to save result. Weights &amp; Biases is better for me than the others (tensorboard, mlflow, etc ...)
