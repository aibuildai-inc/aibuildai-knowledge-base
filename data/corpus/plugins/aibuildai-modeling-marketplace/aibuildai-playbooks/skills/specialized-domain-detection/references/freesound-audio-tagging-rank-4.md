# 4th solution

Competition: freesound-audio-tagging
Rank: #4
Source: https://www.kaggle.com/c/freesound-audio-tagging/discussion/62634

For the solution, we employed both deep learning methods and statistic features-based shallow architecture learners. For single model, only deep learning approaches are investigated, and different deep neural network architectures are tested with different kinds of input, which ranges from the raw-signal, log-scaled Mel-spectrograms (log Mel) to Mel Frequency Cepstral Coefficients (MFCC). For log Mel and MFCC, the delta and delta-delta information are also used to formulate three-channels features. Inception, ResNet, ResNeXt, Dual Path Networks (DPN) are selected as the neural network architectures, while Mixup is used for the data augmentation.

Using ResNeXt, our best single convolutional neural network architecture provides a mAP@3 of 0.967 on the public Kaggle leaderboard,  0.942 on private. Moreover, to improve the accuracy further, we also propose a meta learning-based ensemble method. By employing the diversities between different architectures, the meta learning-based model can provide higher prediction accuracy and robustness with the comparison to the single model. Using the proposed meta-learning method, our solution achieves a mAP@3 of 0.977 on the public Kaggle leaderboard, 0.951 on the private LB.

PS: You can find our code given in
https://github.com/Cocoxili/DCASE2018Task2

Please cite this work in your pulications if it helps your research.

@article{xu2018general,
  title={General audio tagging with ensembling convolutional neural network and statistical features},
  author={Xu, Kele and Zhu, Boqing and Kong, Qiuqiang and Mi, Haibo and Ding, Bo and Wang, Dezhi and Wang, Huaimin},
  journal={arXiv preprint arXiv:1810.12832},
  year={2018}
}
