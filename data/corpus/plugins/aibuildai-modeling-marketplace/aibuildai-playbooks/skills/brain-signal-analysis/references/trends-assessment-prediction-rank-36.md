# 36th Place Solution (with code)

Competition: trends-assessment-prediction
Rank: #36
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162738

Hello everyone, first of all i would like to thank my great teammates @joatom, @mks2192and @kpriyanshu256 for such an interesting competition journey. And congratulations to all the winners!

It's been a great competition, and my team has spend a lot of time in this competition and finally glad to share that all the hard work paid off.


## Our Approach
Our final solution is an ensemble of following models or submissions:
* [simple tabular nn](https://www.kaggle.com/joatom/trends-tabular-nn-0-159) as mentioned by @joatom in the comments below.
* GCN-Tabular Learner by @kpriyanshu256 
* [TReNDS - PyCaret (Training + Inference)](https://www.kaggle.com/rohitsingh9990/trends-pycaret-training-inference) by @rohitsingh9990 
* [RAPIDS SVM on TReNDS Neuroimaging](https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging) by @aerdem4 
* and a bunch of other models trained on tabular data only.

Both [RAPIDS SVM](https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging)and [TReNDS - PyCaret](https://www.kaggle.com/rohitsingh9990/trends-pycaret-training-inference) notebooks are already publicly available, there is nothing much to explain.

### Brief Overview of `simple tabular nn`:
* A detailed explanation of this approach is posted by @joatom in the comments below.
&gt; Note: This is the most important approach which alone when ensembled with public RAPIDS SVR kernel  can easily give us a silver medal.


### Brief overview of our `TReNDS GCN-Tabular Learner`:

* The scenario of using a GCN appears if we treat the correlation values between the fnc entities (eg. SCN(53) ) as edge weights. 
* A little EDA showed that there are ((53*52)/2)  columns (excluding the Id) i.e. a completely connected graph and there are in total 53 unique entities. There are also 53 spatial maps for each patient. 
* We assume `ICN_numbers.csv` gives a mapping of each entity to the corresponding map. Hence, we create a graph of 53 nodes for each patient. To create features for the nodes, we rely on domain specific knowledge ( https://www.kaggle.com/kpriyanshu256/trends-image-features-53-100 ). 
* We extracted some features from each map using nilearn ( https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_atlas_schaefer_2018.html#nilearn.datasets.fetch_atlas_schaefer_2018 and https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_atlas_talairach.html#nilearn.datasets.fetch_atlas_talairach ). 
* We have used this for the GCN. The features from the GCN are collected and combined with the `fnc.csv` and `loading.csv` features. The combined features are passed through dense layers to get the output. 
* Notebook : https://www.kaggle.com/kpriyanshu256/trends-gcn-tab-oof/ (edited) 
kaggle.comkaggle.com


### Post-Process

At last after ensembling, we did a little post-processing that we came up on last day of this competition, which gives us a boost of approx 0.002

```
Before transformation, the age in the training set is rounded to nearest year for privacy reasons. However, age is not rounded to year (higher precision) in the test set. Thus, heavily overfitting to the training set age will very likely have a negative impact on your submissions.
```
This is the information available on the data page, and we were wondering how we can use this sort of information to get best out of it. so, what we did is to try multiplying `age` we get from our ensembled models by a factor of `1.01`

Unfortunately we try this on last day of competition and we didn't have any submissions left to improve it further, i wish we might have tried it earlier.

Link to our final submission kernel https://www.kaggle.com/rohitsingh9990/36th-place-trends-ensemble?scriptVersionId=37747366
