# 31st place solution and some points about the competition

Competition: isic-2024-challenge
Rank: #31
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532593

Since I'm not a computer-related major, the following represents my personal views only, and I welcome exchanges and discussions

1.The gold medal program of ISIC2020 is not better adapted to this competition, I roughly designed about 12 programs based on the last competition transformer, the effect is not ideal!

Personally, I think it may be that my program is not good enough (I made the basic model and data public so that people can make their own combinations), or it may be that the amount of chart data is far greater than the last dataset. (2020: train.csv(2.06 MB), 2024 train-metadata.csv(257.54 MB) )


[https://www.kaggle.com/datasets/aristotlechen/gold-ev2](url)

[https://www.kaggle.com/datasets/aristotlechen/gold-b0-cross128v](url)

[https://www.kaggle.com/datasets/aristotlechen/gold-b0-cross64v](url)

[https://www.kaggle.com/datasets/aristotlechen/gold-bce-b0](url)


2.Since Magic Noise is generated based on the statistical properties of a particular dataset, rather than real features. This causes the model to perform well on the training data (actually overfitting)

https://www.kaggle.com/code/richolson/isic-2024-magic-noise-for-lb-overfit

**But I don't know what happens when one_hot and magic_voice are combined (forgive me for running out of time)，so hopefully one of the users will be able to complete this.**

3.The combination of features requires a lot of experimentation, and simply removing low-importance features does not necessarily improve the CV，So it is better to use ensembles to improve generalization, consider oof+voter+inference models.Adding multiple voters is more useful than increasing the number of layers.

(tip:Personally, I've seen random forests, hgb, etc,ngb model. have better results, but they always time out）

4.Adjusting the parameters to increase the CV but the actual effect is down instead, I tried Bayesian Optimization and Optuna but it didn't work well, maybe only one parameter can be adjusted at a time.

about data drift in detection：

>"Monitoring and Adapting to Concept Drift in Medical Imaging" (2021)
Federica Fornasa, et al.
IEEE Journal of Biomedical and Health Informatics
DOI: 10.1109/JBHI.2021.3062373

>"Automated Detection of Data Drift in Medical Imaging using Deep Learning" (2020)
James Brown, et al.
Nature Machine Intelligence
10.1038/s42256-020-0196-0



>"Continual Learning in Medical Imaging: A Review" (2021)
Matthias Perkonigg, et al.
Medical Image Analysis
DOI: 10.1016/j.media.2021.102115

（It is also possible that there has been data drift, I personally believe that there is some correlation between the probability of disease in the patient group as the age distribution may change, but I have not been trying to consider this as a variable, mainly because of the failure of the variable transformation involved.）


5.**About image processing model selection**

EfficientNet_b0,eva02_small_patch14,edgenext_base

The memory footprint of these three models is more than reasonable, and I've also tried huge and b1, b2, b3,Memory always overflows.

**SelecSLS and NextViT show severe overfitting**

NextViT, as a newer model, may lack extensive validation in some applications and parameter tuning may be difficult，Attentional mechanisms can help the model focus on small but critical distinguishing features.SelecSLS lacks explicit attentional mechanisms such as irregular edges, color variations that can be easily overlooked, especially since the difference between benign and malignant lesions can be very subtle.

*Because of the possible limitations of my knowledge, you are welcome to add to it!*
Last but not least, thank you to all the participants and organizers, everyone shined in the competition!
