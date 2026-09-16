# 9th place solution (Knowledge Distillation)

Competition: imet-2019-fgvc6
Rank: #9
Source: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94837#latest-550728

Hi everyone. This is my solution overview which scored 0.664 on public LB.


### Competition's challenge

1. 9 hours kernels limit for inference. 
2. Lack of proper labels. For example if an image has **culture::american**, there also should be **culture::american or british** but in many cases these labels are lacked. 


### Approach

1. I used **knowledge distillation** to compress the knowledge in an ensemble into student models so that I could use 9 hours more efficiently.
2. Knowledge distillation also attends to the lack of proper labels by providing soft targets to training images. Because competition metric favors recall over precision, this should improve lb.


### Hardware

I used kaggle kernels to train almost all the models. 


### Model training

The model training process can be split into 2 parts. 
- Train **teacher models**
- Train **student models** using outputs of the teacher


### Teacher models

- 2x se\_resnext101
- 2x se\_resnext50

I trained 4 models and averaged their outputs. I shared the performance of them during the competition and you can check that discussion here. https://www.kaggle.com/c/imet-2019-fgvc6/discussion/92159

Averaged output scored **cv 0.6467** and **lb 0.656**. This averaged output(cv 0.6467) on train images were used to train the student models next. 

I did not treat tags and cultures separately.

**training method**
I trained only a dense layer for first 2 epochs because their weights are fresh. I got this idea from the kernel Lopuhin kindly shared and result is better with this.
- 1st epoch (1/10 of baselr, dense layer only)
- 2nd epoch (5/10 of baselr, dense layer only)
- 3rd epoch (1/10 of baselr, all layers)
- 4th epoch (5/10 of baselr, all layers)

**se\_resnext101**
- image size: 320
- optimizer: Adam
- base lr: 1.2e-4
- batch size: 36
- scheduler: ReduceLrOnPlateau or StepLR
- best epoch: 12 to 15
- loss: BCEWithLogitsLoss + FBetaLoss
- augmentations: RandomResizedCrop, Horizontal Flip, Random Erasing
- TTA: RandomResizedCrop, Horizontal Flip (10 times)
- Training time: around 13 hours (Kaggle kernel)

**performance of 5-fold se\_resnext101**
- cv each: 0.627 | 0.629 | 0.627 | 0.628 | 0.630
- lb each: 0.630 | ---
- cv(concat): 0.628
- lb(mean): **0.652**

**se\_resnext50**
- base lr: 1.4e-4
- batch size: 52
- Training time: around 7 hours

**performance of 5-fold se\_resnext50**
- cv each: 0.621 | 0.623 | 0.620 | 0.618 | 0.622
- lb each: 0.621 | ---
- cv(concat): 0.621 
- lb(mean): **0.641**

**augmentations**
```
from torchvision import transforms as T

def train_transform(size):
    return T.Compose([
        RandomResizedCropV2(size, scale=(0.7, 1.0), ratio=(4/5, 5/4)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.3, sh=0.3),
    ])

def test_transform(size):
    return T.Compose([
        RandomResizedCropV2(size, scale=(0.7, 1.0), ratio=(4/5, 5/4)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Student models

- 1x se\_resnext101
- 1x inceptionresnetv2

I trained a new 6-fold se\_resnext101 using outputs of the teacher model on training images. I multiplied it by 0.7 to prevent overtrusting his teacher. I took np.maximum of it and original hard targets to construct new targets. Test images were remained untouched. 

Training parameters remained almost the same from teacher models. This 6-fold model scored **cv 0.668** and **lb 0.662** by itself. 

**performance of 6-fold se\_resnext101 (student)**
- cv each: 0.669 | 0.669 | 0.668 | 0.664 | 0.665 | 0.668
- lb each: 0.649 | 0.648 | ---
- cv(concat): 0.668
- lb(mean): **0.662**

I also trained inceptionresnetv2 and averaged with se\_resnext101 and scored **lb 0.664**. 


### Inference

The inference part is very simple. I just averaged outputs of the student models. # of TTA is 7. 


### Possible Improvements

I have to admit there are lots of space for improvements. 
- Train more teacher models / student models such as senet152, pnasnet for ensembling. My solution definitely lacks variants for better ensembling. 
- Treat cultures and tags separately for training, thresholding as they have different characteristics. 


I'd like to thank Lopuhin for sharing his great kernel where I borrowed some ideas/implementations such as making folds, binarizing outputs. https://www.kaggle.com/lopuhin/imet-2019-submission

I also would like to thank Bac Nguyen and his implementation of FbetaLoss. https://www.kaggle.com/backaggle/imet-fastai-starter-focal-and-fbeta-loss

Thank you kaggle team and Zhang for hosting such a great competition. 

Thanks for reading.
