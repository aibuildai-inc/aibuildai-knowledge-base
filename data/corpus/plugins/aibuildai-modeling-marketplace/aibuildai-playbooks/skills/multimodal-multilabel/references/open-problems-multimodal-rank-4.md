# 4th place solution (with code)

Competition: open-problems-multimodal
Rank: #4
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/366460

## Intro

To begin with, thanks to the Kaggle team and Open Problems team for hosting such a wonderful contest. I would also like to share my gratitude to all the competitors, especially those generous competitors who are willing to share their codes and thoughts like @ambrosm @alexandervc @baosenguo @pourchot and so on. I couldn't have gone so far without their help.

You can find full code on Github [here](https://github.com/oliverwang15/4th-Place-Solution-for-Open-Problems-Multimodal-Single-Cell)

## Cite

### Data preprocessing

At first, all of my feature engineering methods are based on the [original](https://www.kaggle.com/competitions/open-problems-multimodal/data)  data, but my public score raised from 0.812 to 0.813 after I merely change the data source so all the feature engineering processes and based on the [raw](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/359355) data. 

My preprocessing method is using `np.log1p` to change the raw data. I have also tried other preprocessing methods like `MAGIC` and `TF-IDF` but they can't improve my CV score.

### Feature engineering

The final inputs of the models consist of mainly six parts. Three of them are dimension reduction parts including `Tsvd`, `UMAP`, and `Novel’s method`. The rest are feature selection parts including `name importance`, `corr importance,` and `rf importance`.

- `Tsvd`: `TruncatedSVD(n_components=128, random_state=42)`

- `UMAP`: `UMAP(n_neighbors = 16,n_components=128, random_state=42,verbose = True)`

- `Novel’s method`: The original method can be found [here](https://github.com/openproblems-bio/neurips2021_multimodal_topmethods/blob/dc7bd58dacbe804dcc7be047531d795b1b04741e/src/predict_modality/methods/novel/resources/helper_functions.py). At first, I wanted to implement the preprocessing method to replace simple `log1p` but after I replaced the `Tsvd` results of `log1p` by  the `Tsvd` results of `Novel’s method ` I found that my CV went down. But if I kept both of them, the CV score would increase a little bit. So I kept the `Tsvd` results of `Novel’s method `.

- `name importance`: It 's mainly based on AmbrosM's [notebook](https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense/notebook#Name-matching). But I added additional information from `mygene` while matching. I will release my complete preprocessing code later and specific results can be found there.

- `corr importance`: As the name suggested, I chose the top 3 features that correlated with the targets. There was overlap and the number of selected features was about 104

- `rf importance`: Since the feature importances of random forest may apply to NN and other models as well. So I selected 128 top feature importances of the random forest model.

I have also tried other mothed including `PCA`, `KernelPCA`, `LocallyLinearEmbedding`, and `SpectralEmbedding`.`PCA` gives little help and it will cause severe overfitting when used with `Tsvd`. I could' t finish the manifold methods in 24 hours so I gave them up.

### Models

I have implemented the CV strategy like the private test, but it turns out that the strategy like the public test is better. So all of the results are based on `GroupKFold` on `donors`. I have done there-layers stacking in the competition. and I have also done the ensemble on the stacking results and the results of independent models. Here are the models I used and I will also release the code later. 
| Method | Stacking |  NN       |  NN_online  | CNN | kernel_rigde | LGBM    | Catboost |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| CV | 0.89677 | 0.89596 | 0.89580 | 0.89530 | 0.89326 | 0.89270 | 0.89100 |




- `NN`: A personal-designed NN network, trying to do something like the transformers. I used MLP to replace the dot product in the mechanism of attention. This may not be so reasonable and I am also aware of the importance of feature vectors and dot products. But I was so fascinated by attention and I also tried `tabnet` and `rtdl` but they didn't work very well. But my method seemed to work even better than simple MLP. 
  [Demo notebook](https://www.kaggle.com/oliverwang15/4th-solution-cite-nn)

- `CNN`: Inspired by the tmp method [here](https://www.kaggle.com/competitions/lish-moa/discussion/202256) and also added multidimensional convolution kernel like the Resnet. 

- `NN(Online)`: This model is mainly based on pourchot's method [here](https://www.kaggle.com/code/pourchot/all-in-one-citeseq-multiome-with-keras) and only some tiny change was made.
[Demo notebook](https://www.kaggle.com/code/oliverwang15/4th-solution-cite-online-nn)

- `Kernel Rigde `: This model is inspired by the best solution of last year's competition. I used [Ray Tune](https://docs.ray.io/en/master/tune/index.html) to optimize the hypermeters
  [Demo notebook with ray tune](https://www.kaggle.com/code/oliverwang15/4th-solution-ray-tune-krr)

- `Catboost`: There are many options for `catboost` here. Using `MultiOutputRegressor` or `MultiRMSE` as `objective`.But we can't do earlystopping to prevent overfitting in the first method and the result of the second method is not good enough so I made a class `MultiOutputCatboostRegressor` personally, using `MSE` to fit the normalized targets.

- `LGBM`: I also wrote `MultiOutputLGBMRegressor` and the results seem to be better and the training process was so slow that I had to give it up in the stacking. However, I still trained a independent LGBM model and used it in the final training.  
 [Demo notebook](https://www.kaggle.com/code/oliverwang15/4th-solution-cite-multioutputlgbm) 

-  `stacking`: I used `KNN`,`CNN`,`ridge`,`rf`,`catboost`,`NN` in the first layer and only `CNN`,`catboost`,`NN` in the second and just a simple `MLP` in the last layer. To avoid overfitting, I used `KFold` and oof predictions between layers, and every stacking model are using `GroupKFold`(so there are 3 stacking models here). It seems to be a little bit to understand so you may refer to the picture. If you still have confusion please feel free to ask me.
[Demo notebook train](https://www.kaggle.com/code/oliverwang15/4th-solution-cite-stacking-train) 
[Demo notebook predict](https://www.kaggle.com/code/oliverwang15/4th-solution-cite-stacking-predict) 



| CV Results    | Model Ⅰ (vaild 32606) | Model Ⅱ (vaild 13176) | Model Ⅲ (vaild 31800) |
| ---------- | --------------------- | --------------------- | --------------------- |
| Fold 1     | 0.8989                | 0.8967                | 0.8947                |
| Fold 2     | 0.8995                | 0.8967                | 0.8951                |
| Fold 3     | 0.8985                | 0.8959                | 0.8949                |
| Fold Mean  | 0.89897              | 0.89643              | 0.89490                |
| Model Mean | 0.89677              | -                     | -                     |

### Ensemble
[notebook](https://www.kaggle.com/code/oliverwang15/4th-solution-ensemble/notebook) 

## Multi
To be honest, I put most of my efforts on cite part so there is nothing very special here and I will make a brief introduction. 

### Data preprocessing & Feature engineering
#### inputs:
1. TF-IDF normalization
2. `np.log1p(data * 1e4)`
3. Tsvd -> 512
#### targets: 
1. Normalization -> mean = 0, std = 1
2. Tsvd -> 1024
### Models
- `NN`: A personal-designed NN network as mentioned above. The output of the model is 1024 dim and make dot product with `tsvd.components_`(constant) to get the final prediction than use `correl_loss` to calculate the loss then back propagate the grads.

- `Catboost`: The results from online [notebook](https://www.kaggle.com/code/xiafire/lb-t15-msci-multiome-catboostregressor)

- `LGBM`: The same as the `MultiOutputLGBMRegressor` mentioned above. Using `MSE` to fit the tsvd results of normalized targets.

### Ensemble
The same notebook as mentioned above.
[notebook](https://www.kaggle.com/code/oliverwang15/4th-solution-ensemble/notebook)
