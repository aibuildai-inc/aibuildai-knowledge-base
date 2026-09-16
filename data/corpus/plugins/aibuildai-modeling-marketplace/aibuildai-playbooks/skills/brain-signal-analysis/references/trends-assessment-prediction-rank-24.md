# 24th place - 3D fMRI Feature Engineering Partial Solution

Competition: trends-assessment-prediction
Rank: #24
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162844

This was the best competition I joined by far in my opinion. I really enjoyed working on this topic and I might even want to work on neuroimaging as a job. I was little bit disappointed because most of the teams didn't even bother looking at `fMRI_train` and `fMRI_test`. Those teams lost large amount of information by skipping that part of the dataset. Our solution heavily relies on 3D fMRI data and I'm updating my data analysis kernel with all feature engineering code described here. It has more detailed explanation in this kernel here. (It takes 8 hours to create those features so kernel is still comitting)

https://www.kaggle.com/gunesevitan/trends-neuroimaging-data-analysis-3d-features

## 3D Feature Engineering

Features are based on the differences of fMRIs between different target groups. The biggest differences in those fMRIs are the spread and intensity of active/inactive areas. I plotted the youngest and oldest subject's fMRIs side by side for 53 components, and tried to extract some features. Some of those features worked, most of them didn't. 



Data in `fMRI_train` and `fMRI_test` are represented in 4 dimensional space: 3 spatial dimensions and one component dimension. The 4th dimension is time in most cases but it is 53 different components in this dataset. 3D brain is mask is applied to 4D samples in order to convert them into a restructured 2D representation.



```
fmri_mask = nl.image.load_img('../input/trends-assessment-prediction/fMRI_mask.nii')
sample = h5py.File(f'../input/trends-assessment-prediction/fMRI_train/10001.mat', 'r')['SM_feature'][()]
sample = np.moveaxis(sample, [0, 1, 2, 3], [0, 3, 2, 1])
masked_sample = sample[:, fmri_mask.get_data() == 1].astype(np.float32)
```
This code loads the fMRI mask `(53, 63, 52)`  and a random sample `(53, 52, 63, 53)`. Reorients the axes of sample `(53, 53, 63, 52)` and apply the mask to it. Final shape of the sample is `(53, 58869)` which is `(n_components, n_voxels)`. All of our features are extracted in this space. The same fMRIs look like this when they are masked.

 

The features created in this space are:
* Skew and kurtosis of every 53 components
* Skew of `np.diff(component)`
* Active and inactive regions are created for 53 components with the lines below, and their skew and shapes are extracted
  *  `active = component[component &gt; component_mean + component_std]`
  * `inactive = component[component &lt; component_mean - component_std]`
* Active mean / inactive mean and active std / inactive std are created for 53 components
* Component skew - active skew and component skew - inactive skew are created for 53 components
* More active and inactive regions are created for 53 components with the lines below, and their shapes are extracted
  * `more_active = component[component &gt; component_mean + (2 * component_std)]`
  * `more_inactive = component[component &lt; component_mean - (2 * component_std)]`
* Final features are each component (signal) are divided into 15 equal parts which corresponds to bounding boxes in 3D space. Active region shape and mean of each box are extracted with the lines below. I intended to capture active region spreads and intensities in fixed locations with those features.

 ```
box_count = 15
for b, box in enumerate(np.array_split(component, box_count)):
    features[f'{icn_order[i]}_box{b}_mean'] = box.mean()
    box_active = box[box &gt; component_mean + component_std]
    features[f'{icn_order[i]}_box{b}_active_spread'] = box_active.shape[0],
```

At the end I created 2279 new features and they helped my not so special models to reach 0.1581.
```
Number of Bounding Box Features: 1590
Number of Statistical Features: 689
```

I also tried smoothing with Gaussian filter before feature engineering but it either had no effect or lost too much information as seen below. (First image has no smoothing, second image has smoothing with factor 5 and third image has smoothing with factor 10)



## Models

My models were very simple. I used 2 ridge regressions and 2 rapids SVRs. Two of those models included FNC features and two of them didn't. I used 10 KFold with shuffle as cross-validation and trained each model with 5 different shuffle seeds. I also tried to drop 5% of FNC features and bounding box features in every different iteration for diversity. Finally, I added ±0.1 random noise to `age` for increasing cardinality since it was rounded. Models are shared in this kernel here.

https://www.kaggle.com/gunesevitan/trends-neuroimaging-linear-model-ensemble
