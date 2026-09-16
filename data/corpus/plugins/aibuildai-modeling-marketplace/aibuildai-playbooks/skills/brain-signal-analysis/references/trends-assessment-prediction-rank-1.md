# 1st private (1st public) place + code

Competition: trends-assessment-prediction
Rank: #1
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/163017

Our team: Nikita Churkin @churkinnikita, Dmitry Simakov @simakov.

First of all, thanks to the organizers, Kaggle team and all participants: it was important and very exciting challenge for our team. In addition, we want to thank everyone who wished us luck in kaggle discussions.

The core of our approach is:
I. Generate powerful features from 3D fMRI data.
II. Train-test “matching”.
III. Training of a diverse ensemble (different models and feature subsets).
IV. Сombining all together.

Let us describe the details.

## I. Generate features from 3D fMRI data.

We tried to use 3D CNN models at first, but did not spend so much time on it.
 Small network (resnet10 without pretraining) with AdamW, L1 regularization, [ard](https://github.com/HolyBayes/pytorch_ard) layers, snapshots and CutOut3D augmentation worked better as a regressor. 3D autoencoders (simple or UNet-like) + PCA (or ICA) also worked, but was slightly worse for us. 

The score overall was quite low (only age and domain2-var2 were higher than median prediction) and composing with others features was difficult (3D CNN model trains 10 epochs, whereas MLP on loadings and fnc 70 epochs). So we (mostly @simakov) decided that it was too hard problem for many of top20 participants and they must use something different (it is great that we are wrong, your solutions are cool and interesting!).

Our research showed that (obviously) the most important problem was high dimensionality of 3D data. Most related papers use some dimensionality reduction methods to solve this problem. We already had data after such procedure (see data description). But we wanted to move further. If we understood right, data preparation pipeline consisted of two stages: intra-subject level and inter-subject level (it will be cool to hear more about it from @miykael). For the second one, we had to work with all subjects, but they did not fit in the memory. Nipype probably can work with such data, but it is too complicated for fast usage (and @miykael great notebooks don't help ). So, we used classics – sklearn! Sklearn has some algorithms for mini-batch dimensionality reduction: dictionary-learning and Incremental PCA with useful method ```partial_fit```. 
In theory, we prefer to use as much components as possible (there is probably a lot of information in 3D scans), use as much features (channels) as possible (for better interactions among them), and have smallest batch-size (RAM limit). This algorithms have condition that bath-size &gt;= n_components.

The most useful scheme was Incremental PCA with n_components 200, batch-size 200. We splited channels in groups by 10 and flattened inside them (6 groups in total). All training took 9 hours for 24 threads processor and 64gb RAM. As a result, we had 1200 PCA features and they improved our model by 2.3e-3. We also tried dictionary-learning for diversity with n-components 100, with batch-size 100 and n-iters 10 (training procedure took 2 days). We fitted models on train + test data without any additional preprocessing (different settings were generating lower scores or asked for much more RAM). 

We refused to use 3D CNN models in our submissions – it was too long and we already saw great score on the CV and Public LB.

In addition, we computed some simple statistics, like mean, std, and quantiles inside each feature of 3D scans and similar stats for FNC matrix.

## II. Train-test matching
### Preprocessing for constructed features.

The idea is actually simple: add a bias to different columns in test set to make it closer to train set. Linear models showed good performance in the competition so it was expected that adding biases would help a lot (at least for linear models).  There are lots of way to figure out possible biases: we used minimization of Kolmogorov-Smirnov test’s statistic between train[col] and test[col]+ b.


 
You can see that resulting distributions are not perfectly matched but all gaps mostly disappeared: it was enough to boost our public/private score significantly for any single model or the whole ensemble. We used biases for site2 data (known points + classifier output) and separate biases for other test data.

The same logic can be applied (as postprocessing) to submission.

### Design of good site2 classifier
At the beginning, we thought that classifier was the most important part for surviving shakeup. As it was mentioned before, offsets for site2 (both known and with classifier) improved the public LB score by 1е-4. The same effect they had in the private LB. So, we can took the first place even without classifier and offsets for site2.

Nevertheless, our approach:

Raw loading, fnc and pca features were used for site2 classifier. We applied StandartScaler for train + test data (it boosts score by ~0.02 roc-auc). Regression ElasticNet was used for modeling. Then, we made pseudo-labes inside training fold (hard labels for only site2 with threshold 0.3) and retrained model with pseudo-labels (only +0.004 gain). Final metrics were: 0.973 roc-auc, 0.76 F1-score (subject to site2 class). Our model detected ~1400 new site2 observations, but 200-400 of them were False Positive errors.



If we apply offsets before training classifier, roc-auc becomes 0.67 and our model poorly distinguishes site1 form site2.

### Postprocessing. 
Postprocessing used the same logic as a preprocessing: we try to make the distribution of site2 predictions (revealed + from classifier) closer to distribution of site1 predictions. We calculate KS statistic for site2 and not site2 predictions and find best shift. But the effect of postprocessing was small: only 5e-5 for both public and private. 

## III. Training of a diverse ensemble (different models and feature subsets).
Main and only “model” for domain1_var2 was median prediction.

And for other targets we used ML models to create "score" for every label and every feature subset. Scores are just OOF predictions on the data of only one source (like, for example, FNC features or dictionary learning features). Random Forests and GBM performed poorly in making scores, learning on raw features or stacking.

But we wanted to use something tree-based in our ensemble (to add diversity) and recalled about [RGF ](https://arxiv.org/abs/1109.0887).

RGF showed nice performance in construction of scores and all levels of stacking (especially for age prediction).

Making of scores allowed us to effectively use more sophisticated and/or slow algorithms (like MLP or RGF) on top of, in some way, all data at once. Scores were build by blending some fitted linear models (like Ridge or OMP) and RGF.

Our stacking approach involved different models on different datasets (including scores) for each target. Final stacking for complicated labels such domain2-var1 and domain1-var1 included only blend of basic linear models + NuSVM, KernelRidge and Gaussian Process trained on different datasets. Final ensemble of domain2-var2 contained more models but nothing truly different from complicated case (just a little more of dataset-model packages).

Because detailed listing of all target-dataset-model is boring we are going to explain detailed stacking scheme only for age.


 

## TLDR:
1) Incremental PCA for 3d images.
2) Offsets for test features (like we did in ION).
3) Large (4-5 levels) stack.
Our path can be summarised in that plot:



UPDATE:
Our code: https://github.com/DESimakov/TReNDS/
