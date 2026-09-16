# 1st place solution /w code

Competition: bengaliai-cv19
Rank: #1
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/135984

# 1st Place Solution --- Cyclegan Based Zero Shot Learning





## Classes and Labeling

I did not make inferences about the parts of the character. In other words, all my models classify against the 14784 (168  * 11  * 8) class.
Therefore, I needed to know what the combination of labels made up of Grapheme.

I predicted the relationship between the combination of labels and Grapheme from the label of the given train data, and created the following code.

```python=

class_map = pd.read_csv('../input/bengaliai-cv19/class_map.csv')
grapheme_root = class_map[class_map['component_type'] == 'grapheme_root']
vowel_diacritic = class_map[class_map['component_type'] == 'vowel_diacritic']
consonant_diacritic = class_map[class_map['component_type'] == 'consonant_diacritic']
grapheme_root_list = grapheme_root['component'].tolist()
vowel_diacritic_list = vowel_diacritic['component'].tolist()
consonant_diacritic_list = consonant_diacritic['component'].tolist()

def label_to_grapheme(grapheme_root, vowel_diacritic, consonant_diacritic):
    if consonant_diacritic == 0:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root]
        else:
            return grapheme_root_list[grapheme_root] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 1:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]
        else:
            return grapheme_root_list[grapheme_root] + vowel_diacritic_list[vowel_diacritic] + consonant_diacritic_list[consonant_diacritic]
    elif consonant_diacritic == 2:
        if vowel_diacritic == 0:
            return consonant_diacritic_list[consonant_diacritic] + grapheme_root_list[grapheme_root]
        else:
            return consonant_diacritic_list[consonant_diacritic] + grapheme_root_list[grapheme_root] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 3:
        if vowel_diacritic == 0:
            return consonant_diacritic_list[consonant_diacritic][:2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic][1:]
        else:
            return consonant_diacritic_list[consonant_diacritic][:2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic][1:] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 4:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]
        else:
            if grapheme_root == 123 and vowel_diacritic == 1:
                return grapheme_root_list[grapheme_root] + '\u200d' + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]
            return grapheme_root_list[grapheme_root]  + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 5:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]
        else:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 6:
        if vowel_diacritic == 0:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]
        else:
            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]
    elif consonant_diacritic == 7:
        if vowel_diacritic == 0:
            return consonant_diacritic_list[2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[2][::-1]
        else:
            return consonant_diacritic_list[2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[2][::-1] + vowel_diacritic_list[vowel_diacritic]
```


The generation of synthetic data and the conversion of the prediction results into three components are based on this correspondence.

## Split Data for Unseen Cross Validation


All are randomly selected and split.Ashamedly, I split the data without thinking, so that a non-existent Grapheme root class was created at the time of evaluation, and proper evaluation could not be performed.
However, since the learning cost was very high, it could not be easily recreated, and all local cv were evaluated as they were.
😭


## (1) Out of Distribution Detection Model

It is a model for distinguishing whether the input image is a Seen class or an Unseen class.
This model outputs confidence for each of the 1295 classes independently. If all confidences are low, it is judged as Unseen, and if there is at least one confidence, it is judged as Seen class.


- No resize and crop
- Preprocess --- AutoAugment Policy for SVHN ([https://github.com/DeepVoltaire/AutoAugment](https://github.com/DeepVoltaire/AutoAugment))
- CNN --- EfficientNet-b7(ImageNet Pretrained)
- Optimizer --- `torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)` use defaule value
- LRScheduler --- WarmUpAndLinearDecay
```python=
def warmup_linear_decay(step):
    if step &lt; WARM_UP_STEP:
        return step/WARM_UP_STEP
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)
```
- Output Lyaer --- LayerNorm-FC(2560 -&gt; 1295)-BCEWithLogitsLoss + OHEM?
```python=
        out = model(image)
        sig_out = out.sigmoid()
        loss = criterion(out, one_hot_label)
        train_loss_position = (1-one_hot_label)*(sig_out.detach() &gt; 0.1) + one_hot_label
        loss = ((loss*train_loss_position).sum(dim=1)/train_loss_position.sum(dim=1)).mean()
```
- Epoch --- 200
- Batch size --- 32
- dataset split --- 1:0
- single fold
- Machine Resource --- 1 Tesla V100 6 days



| 1168 class Local CV(auroc) |
| -------- |
| 0.9967         |


## (2) Seen Class Model

This model classifies 1295 classes included in the training data.

- No resize and crop
- Preprocess --- AutoAugment Policy for SVHN ([https://github.com/DeepVoltaire/AutoAugment](https://github.com/DeepVoltaire/AutoAugment))
- CNN --- EfficientNet-b7(ImageNet Pretrained)
- Optimizer --- `torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)` use defaule value
- LRScheduler --- WarmUpAndLinearDecay
```python=
def warmup_linear_decay(step):
    if step &lt; WARM_UP_STEP:
        return step/WARM_UP_STEP
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)
```
- Output Lyaer --- LayerNorm-FC(2560 -&gt; 14784)-SoftmaxCrossEntropy
- Epoch --- 200
- Batch size --- 32
- dataset split --- 9:1 random split
- single fold
- Machine Resource --- 1 Tesla V100 6 days




| CV Score | LB Score |
| -------- | -------- |
| 0.9985     | 0.9874     |

Large gaps can be predicted due to the presence of Unseen Class.

## (3) Unseen Class Model

Learning this model is done in two stages. The first step is training a classifier for images synthesized from ttf files. The second step is training the generator that converts handwritten characters into the desired synthesized data-like image. To perform these learnings, first select one ttf and generate a synthetic dataset.
The image size of the synthesized data was 236x137, the same as the training data. Using Pillow and raqm, I drew Graphheme in four sizes, `[84, 96, 108, 120]`. The size of the dataset is 59136.



### Font Classifier Pre-training


- crop and resize to 224x224
- preprocess --- random affine, random rotate, random crop, cutout
- CNN --- EfficientNet-b0
- Optimizer --- `torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)` use defaule value
- LRScheduler --- LinearDecay
```python=
WARM_UP_STEP = train_steps*0.5

def warmup_linear_decay(step):
    if step &lt; WARM_UP_STEP:
        return 1.0
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)
```
- Output Lyaer --- LayerNorm-FC(2560 -&gt; 14784)-SoftmaxCrossEntropy
- Epoch --- 60
- Batch size --- 32
- Machine Resource --- 1 Tesla V100 4 hours


### CycleGan Training



- crop and resize to 224x224
- preprocess --- random affine, random rotate, random crop (smaller than pre-training one), and no cutout
- Model --- Based on [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) + Pre-trained Font Classifier(fixed parameter and eval mode)
- Optimzer --- `torch.optim.Adam(params, lr=0.0002, betas=(0.5, 0.999))`
- LRScheduler --- LinearDecay
```python=
WARM_UP_STEP = train_steps*0.5

def warmup_linear_decay(step):
    if step &lt; WARM_UP_STEP:
        return 1.0
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_optimizer, warmup_linear_decay)
discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(discriminator_optimizer, warmup_linear_decay)
```
- Epoch --- 40
- Batch size --- 32
- Machine Resource --- 4 Tesla V100 2.5 days
- HyperParameter --- lambda_consistency=10, lambda_cls=1.0~5.0


Please see the paper for CycleGan. The points that are different from normal CycleGan are as follows. The discriminator adds the supervised loss of the pre-trained font classifier when calculating the loss of a handwritten2font generator.

The CV of the one that gave the highest score (lambda_cls = 4.0) is as follows. To avoid the influence of non-existent Graphheme root classes, macro average recall is calculated after excluding non-existent classes. (It does not mean that the recall value of a class that does not exist is calculated as 0.0 or 1.0.)



| Local CV Score for Unseen Class | Local CV Score for Seen Class    |
| ------------------------- | --- |
| 0.8804                    |   0.9377  |

After performing hyperparameter tuning, I trained two other types of ttf without evaluating local cv. They gave a higher LB score than the model trained in the first ttf, so the CV score of the parameter used in private may be higher.


I created two models using different ttf for submission. At the time of submission, inferences were made using these ensembles.

-   [https://www.omicronlab.com/download/fonts/kalpurush.ttf](https://www.omicronlab.com/download/fonts/kalpurush.ttf)
-   [https://www.omicronlab.com/download/fonts/NikoshLightBan.ttf](https://www.omicronlab.com/download/fonts/NikoshLightBan.ttf)

CycleGan's training code will be released after being modified so that it can run on 1GPU. Please wait.


## Strange Points

To be honest, I can't justify why this method would increase the generalization of the Unseen class.
I expected that the generation of images very similar to the synthetic data would affect Consistency Loss and Discriminator and gain generalization to the Unseen Class. However, as the hyperparameters were adjusted, it was observed that the generalization performance improved while the image became unnatural.




| lambda_cls | Input Image | Generated Image | True Synthesis Image | Local CV |
| ---------- | ----------- | --------------- | -------------------- | -------- |
| 1.0        |             |      | |     0.8618     |
|    4.0        |        |         |                 |     **0.8804**     |


## Code

The reduced version of CycleGAN code is now available.

1. https://www.kaggle.com/linshokaku/cyclegan-classifier
   Font image classifier training code (7h)
2. https://www.kaggle.com/linshokaku/cyclegan-training
   Training code for a generator that converts a handwritten image to a font image (8h)
3. https://www.kaggle.com/linshokaku/cyclegan-submission
   Submission code

Running from 1 to 3 in order is equivalent to the method I used in this competition.
Due to execution time and hardware resources, the batch size, training epoch, etc. are set very small, but if you run it with the hyper parameters as mentioned above, you should get results comparable to my execution results.
I hope you find it helpful.
