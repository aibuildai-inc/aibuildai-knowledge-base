# 5th place solution, 3D fMRI part

Competition: trends-assessment-prediction
Rank: #5
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162742

Fisrst of all, many thanks to Kaggle and the hosts for hosting such an interesting competition, and congratulations to all the winners. And a special thanks to all my teammates for teaming up with me!! I arrived at this solution with a lot of advice from my teammates @masatomatsui @kurupical @takoihiraokazu .

(Let me apologize for my poor English. )


# Summary of models.
.png?generation=1593476631685161&amp;alt=media)


ResNet, SEResNet based on: https://github.com/kenshohara/3D-ResNets-PyTorch/tree/540a0ea1abaee379fa3651d4d5afbd2d667a1f49
DenseNet: https://github.com/Project-MONAI/MONAI
Other tips: https://www.kaggle.com/c/trends-assessment-prediction/discussion/147797
(Thanks, @shentao !!)


# About Scores

All of 3D models are not so good about CV/LB score. Most of the models are around 0.171 ~ 0.175 CV scores, even the best model can only gets CV 0.17074 (SEResNet18 with resize, float 32).
However, ensembling these models greatly improved the scores. In fact, weighted average of single 3D CNN model(ResNet18, float 32, no normalization) (LB 0.1675) and kernel prediction of [“RAPIDS Ensemble for TReNDS Neuroimaging”](https://www.kaggle.com/tunguz/rapids-ensemble-for-trends-neuroimaging) (LB 0.1595) at 0.2 : 0.8 gots LB 0.1582.
In the end, stacking our various models greatly improved the LB score. 

# About 3D Data

In the early part of the competition, I saved 3D data in float 16 format (light blue model). This is because it made the learning process so much faster and allowed me to train a lot of models. Training one model (5 folds, 10 epochs each) took about 5 ~ 10 hours to complete. In case of float 32, it took 2 ~ 3 times longer.
However, it would be more accurate to train in float 32 format, so important red parts were trained with float 32 format.

# About Resize

To create a variety of model prediction, I resized 3D data from (50, 63, 53) to (75, 94, 79). This contributed to improved CV scores and ensembles.

# About Site Normalization

There were two types of data in the test dataset: site 2 and not revealed(site 1 or site 2). Teammates figured out that, by moving the distribution of site 2 closer to that of site 1 (this is “shift” in the figure), models using loading and fnc gots better LB scores (for example, best kernel 0.1590 -&gt; 0.1586). To achieve something similar in the 3D CNN models, I normalized 3D data by mean, std of each site. Site 1 data by mean, std of site 1, site 2 by those of site 2.
One of my teammates made a model to predict whether the data is site 1 or 2 (Accuracy over 0.94), and I used his predictions to normalize the not reveal data in the same way.
This “Site Normalization” did not boost CV, but improved LB scores.

&gt; DenseNet, float 16, no resize
No normalization: CV 0.1722 -&gt; LB 0.1684
Site normalization: CV 0.1742 -&gt; LB 0.1680
