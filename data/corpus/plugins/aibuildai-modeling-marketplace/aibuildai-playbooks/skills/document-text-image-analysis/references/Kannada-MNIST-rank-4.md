# 4-th place solution: Over9000 optimizer

Competition: Kannada-MNIST
Rank: #4
Source: https://www.kaggle.com/c/Kannada-MNIST/discussion/122430

Congratulations to all participants. The main goal for me in this competition was checking the effective way of using new optimizes like [RAdam](https://arxiv.org/pdf/1908.03265.pdf) and [Over9000](https://github.com/mgrankin/over9000). Despite those optimizers are promised to work much better than Adam in corresponding studies, my initial tests of these optimizers were quite disappointing: RAdam gave about the same result as Adam while Over9000 worked much worse. In the last competition on Cloud Segmentation I participated in gave quite interesting observation: **if RAdam is used there must be no warm-up in cosine annealing**.  Over9000 worked even better with such scheduler than RAdam as expected from [here](https://github.com/mgrankin/over9000). With fast.ai it can be done with the following additional arguments `learn.fit_one_cycle(36, max_lr=slice(0.2e-2,1e-2), pct_start=0.0, div_factor=100)`. So, this competition has provided quite small dataset and fast training to experiment with optimizes and schedulers.

In this competition I used 3 models:
1) **Resnet20** from [CIFAR-pretrained-models](https://github.com/chenyaofo/CIFAR-pretrained-models) (and the corresponding [dataset](https://www.kaggle.com/iafoss/cifarpretrainedmodels)) on 32x32x1 images (I have replaced first conv summing corresponding pretrained weights).
2,3) Pretrained **DenseNet121** and **DenseNet169** on 64x64x1 images (because of the first layer with stride 2 conv followed by 3x3 pooling it is really helpful to upscale small images).

In all models ReLU is replaced by **Mish**. The head is similar to one used by fast.ai by default: Concat pooling + Mish + BN + Dropout(0.5) + Linear + Mish + BN + Dropout(0.5) + Linear.


**Other things used**:
- **Over9000 one cycle without warm-up scheduler**
- Categorical Cross Entropy
- gradient clipping
- best model selection
- discriminative lr with split of the model into backbone and head
- standard fast.ai augmentation without flip `get_transforms(do_flip=False,max_zoom=1.2)`
- 5 fold CV and ignore public LB


**Things that could work**:
- PL could easily boost the private LB score to 0.995+
- More models


**Things that didn't work**:
- ArcFace loss (I expected that it will make the class representation more compact and robust to train/test data mismatch)
- Backbone freezing and training the head first (even without replacement of ReLU by Mish)
- ResNet44 and ResNet56 from CIFAR-pretrained-models worked worse than ResNet20
- use Dig-MNIST

Regarding train/test mismatch: I expect that each person has provided several handwritten digits. The leak happens when digits written by the same person are split between train and val. I tried to assume that each person has provided one or several (fixed number) instances of digits. It didn't work for me because most likely people has provided the different number of handwritten examples, and everything still remains mixed.
RAdam and Over9000 could be added to your kernels with this [utility script](https://www.kaggle.com/iafoss/radam).
