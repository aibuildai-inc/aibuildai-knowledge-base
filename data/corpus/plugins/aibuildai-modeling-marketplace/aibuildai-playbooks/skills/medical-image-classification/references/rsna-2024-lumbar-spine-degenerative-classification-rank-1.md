# 1st place solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #1
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091

First of all, I would like to express my sincere gratitude to the competition host and the Kaggle staff for organizing such a fascinating competition. I thoroughly enjoyed this competition and learned a great deal in the process!

Furthermore, I'd like to thank @hengck23 and @brendanartley. @hengck23 's discussion and notebook were the starting point for my solution, and @brendanartley 's [this dataset](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/524500) helped my coordinate prediction models. I was deeply impressed by their contributions to the Kaggle community.

This is my first solution write-up, so please feel free to leave any comments or suggestions for improvement!

## Summary

My solution is 2 stage approach, creating `test_label_coordinates.csv` and predicting severity. Furthermore, I separated 1st stage into instance_number prediction and coordinate prediction. Therefore I prepared 3 type of model, instance_number prediction model, coordinate prediction model and severity prediction model. The pipeline is shown in the following figure. 

[pipeline]

## 1st stage: test_label_coordinates creation

In the 1st stage, I use 2 type of models, 3D convolution model and 2D convolution model. These models are very simple, encoder + level-separated heads. 

### instance_number prediction (sagittal)

In this part, I used simple 3D ConvNeXt to predict instance_number for each level. Data that is fed into models is just normalized from 0 to 1, sorted by dicom's metadata and padded 32 to depth direction to align shape. Data preprocessing is shown in the following figure (scs example). 

[scs_volume_example]

In training models, I trained models 2 tasks, regression and classification, and I used L1 Loss and Cross Entropy Loss respectively. In the classification task, these heads output (bs, 32) shape logits for each level. In the regression task, these heads output (bs, 3) shape vectors for each level. (bs, 3) shape vector means (x, y, z) and I used z for depth prediction, (x, y) were used auxiliary loss. In the regression task, I normalized coordinate labels 0 to 1 for stabilizing models during training. Concretely, I used label (x', y', z') = (x/width, y/height, z/32). The model architecture is shown in the following image (scs example). I implemented 3D ConvNeXt for this task (to implement 3D ConvNeXt, I referred to [this repo](https://github.com/FrancescoSaverioZuppichini/ConvNext)). 

[instance_number_prediction_scs_example]

The results of instance_number prediction models are shown in the following table (sagt2, scs). 

| model/error | +-0 | +-1 | +-2 | error>+-2 | 
| --- | --- | --- | --- | --- |
|cls| 71.08% | 27.04% | 1.43% | 0.44% |
|reg| 67.48% | 30.59% | 1.61% | 0.31% |


I ensembled this 2 type of predictions using median for each level (actually I used 5 fold for each task). 

### coordinate prediction(sagittal)

In coordinate prediction task, I used 2d encoder + level-separated heads, almost same as instance_number regression model. Data is 3 channel image. The image is picked up using median of instance_number of L1 ~ S1. Then the data processed normalization and reshaping (512x512). Labels are (x', y') = (x/width, y/height)
for each level, same as instance_number regression, and also I used L1 loss. The model architecture is shown in the following figure. 

[coordinate_prediction_model_scs_example]

I used ConvNeXt-base and Efficientnet-v2-l for this task. Before I train these models, I trained these models using @brendanartley 's [dataset](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/524500). These pretrained models were slightly better than pretrained models that were trained using imagenet. I ensembled these predictions using mean. 

### instance_number calculation and coordinate prediction (axial)

For instance_number prediction of axial, I borrowed @hengck23 's method (notebook is [here](https://www.kaggle.com/code/hengck23/2d-to-3d-projection-for-dicom/notebook)). Then I predicted coordinates of axial, same as coordinate prediction for sagittal. 

## 2nd stage: severity prediction

For the 2nd stage, I attempted simple 2.5D model and MIL. 2.5D model can be implemented easily, however, MIL was better than simple 2.5D at final. 

## preprocessing

### Cropping method

My preprocessing strategy is cropping. For example, I cropped sagt2 image for scs; 

1. pick up 5 images (center is an image that was assigned instance_number)
2. reshape 512x512
3. crop images using the coordinate (96 pix left and 32 pix right from coordinate x, 40 pix upper and 40 pix lower from coordinate y)

After cropping an image, the image can be like the figure below (sagt2 for scs, L1/L2). 

[scs_cropped_image]

sagt2, sagt1 and axial were cropped for each classification task. The following tables are representing cropping range from (x, y) coordinate. 

**for scs**

| type | left | right | upper | lower |
| --- | --- | --- | --- | --- |
| sagt2 | 96 | 32 | 40 | 40 |
| axial | 96 | 96 | 96 | 96 |

Note that when I crop images from axial, I picked up left or right  subarticular stenosis coordinate randomly, and for adjusting cropping point, I added +-20 to ss coordinate x. As a result, cropping range can be like the following figure (the example is right ss coordinate x + 20). 

[axial_for_scs_cropping]

**for nfn**

| type | left | right | upper | lower |
| --- | --- | --- | --- | --- |
| sagt1 (both left and right)| 96 | 64 | 32 | 32 |
| axial (right) | 144 | 48 | 96 | 96 |
| axial (left) | 48 | 144 | 96 | 96 |

**for ss**

| type | left | right | upper | lower |
| --- | --- | --- | --- | --- |
| axial (right) | 144 | 48 | 96 | 96 |
| axial (left) | 48 | 144 | 96 | 96 |

The following image is the range of cropping axial for right subarticular stenosis. 

[axial_cropping_for_ss_right]

### data augmentations

I used several augmentations like below; 

*Before cropping*

* random shift of coordinate x and y (-10~+10 pix)
* random shift of instance_number (-2~+2. shifting probability was decided error probability of each instance_number prediction models)

*After cropping*

* RandomBrightnessContrast(p=0.25)
* ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.1, 0.1), rotate_limit=20, p=0.5)

Especially, random shift of instance_number was crucial for robustness of error of 1st stage. 


## model architecture

My model architectures are shown in following figures. 
**[EDITED]**  I have updated the figure illustrating the model architecture to correct an error in the previous version. `aux_attn_score` in the code below is fed into cross entropy loss directly. 

[severity_prediction_model_scs_fixed]
[severity_prediction_model_ss_fixed]

I used ConvNeXt-small and Efficientnet-v2-s as the encoder. After implementing Attention-based MIL, my public LB score was improved from 0.37 -> 0.35. Then, adding bi-LSTM, aux losses and ensembling improve my score from 0.35 to 0.33. bi-LSTM + Attention-based MIL was implemented like below. 

```python
class LSTMMIL(nn.Module):
    def __init__(self, input_dim):
        super(LSTMMIL, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim//2, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.aux_attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    def forward(self, bags):
        batch_size, num_instances, input_dim = bags.size()
        bags_lstm, _ = self.lstm(bags)
        attn_scores = self.attention(bags_lstm).squeeze(-1)
        aux_attn_scores = self.aux_attention(bags_lstm).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_lstm).squeeze(1)

        return weighted_instances, aux_attn_scores
```

## what didn't work

* MAMBA and Self-Attention instead of bi-LSTM
* sharing weight between aux_attention layer and attention layer
* sagt1 image for scs, sagt1 and sagt2 image for ss, sagt2 image for nfn
* long epochs (I used 7 epochs for convnext-small and 14 epochs for efficientnet-v2-s)
* large models (convnext-large < convnext-base < convnext-small in my experiments)
* vision transformers (I think this was my problem. but convolution models were better than vits in my experiments)

## code

All training code is implemented in google colaboratory. All models are used for [this inference code](https://www.kaggle.com/code/wadakoki/rsna-infer-pipeline-public/notebook). Following links are pairs of model name & training notebook link. You can check these model name in the [inference code](https://www.kaggle.com/code/wadakoki/rsna-infer-pipeline-public/notebook). 

You can train on google colaboratory environment with T4 + high memory. 

- [models](https://www.kaggle.com/datasets/wadakoki/rsna-spine-final-models/data)

### instance number prediction models (SCS)

- scs_depth_1024_ssr: [notebook](https://colab.research.google.com/drive/1JbSFgIwxlviyXb6uHv4vfbyqyjbdCRkw?usp=sharing)
- scs_depth: [notebook](https://colab.research.google.com/drive/19YylaxYLYk1q6IOHpfi9UMnhbNUQBYML?usp=sharing)
- scs_depth_1024_ssr_l1: [notebook](https://colab.research.google.com/drive/11fV56U5hPL2IiRzxaiLmuwyjCrygWsgO?usp=sharing)

### instance number prediction models (NFN)
- nfn_depth_1024_ssr: [notebook](https://colab.research.google.com/drive/1sIU9Aun1S1vla_W-4fZ24tGcn-IFD0a_?usp=sharing)
- nfn_depth: [notebook](https://colab.research.google.com/drive/1EPcR7F5p2SvcgpaJdNKw-0vYyc7eYjwr?usp=sharing)
- nfn_depth_1024_ssr_l1: [notebook](https://colab.research.google.com/drive/1Gk4Db4tjhxUEL3uSLiRVlG1MeeGx6K6l?usp=sharing)

### coordinate prediction models (SCS)

- scs_detect_pre: [notebook](https://colab.research.google.com/drive/1qIXQRLkLFyXzyvP9jA2gaX6YZ4_qta27?usp=sharing)
- scs_detect_pre_effv2l: [notebook](https://colab.research.google.com/drive/18vB2qrrBxC7Q4dwVQDR-ioPOY46oEbfu?usp=sharing)

### coordinate prediction models (NFN)

- nfn_detect_pre: [notebook](https://colab.research.google.com/drive/1IPiJDgPDXxOqNTbppZzM89n3ZuPKeWiV?usp=sharing)
- nfn_detect_pre_effv2l: [notebook](https://colab.research.google.com/drive/1eKkZFqKUWZIacJrJYGM1PswYYSi3ztjx?usp=sharing)

### coordinate prediction models (SS)

- ss_detect: [notebook](https://colab.research.google.com/drive/1J3Pj8RMbDm5mG0vvBztyrSRK4G8-NLnU?usp=sharing)

### severity prediction models (SCS)

- _scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_convnext-s_for_exp: [notebook](https://colab.research.google.com/drive/1dWJUGhubs067mJ0GaIOZn8xUt_-1507_?usp=sharing)
- _scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_effv2s_for_exp: [notebook](https://colab.research.google.com/drive/1SsqZOCv7eSYZfqcu5V6ufsHbPp3yN94X?usp=sharing)

### severity prediction models (NFN)

- nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_convnext-s: [notebook](https://colab.research.google.com/drive/1GtC4vtGVo2sY1Mr6cZf7J1IltO2poAEW?usp=sharing)
- nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_effv2s: [notebook](https://colab.research.google.com/drive/1PJ5qU6szTPagEWCMXBYZ3WZPzK5L6_SJ?usp=sharing)

### severity prediction models (SS)

- ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_effv2s: [notebook](https://colab.research.google.com/drive/1CUmMoQeUy2ataJoDffEHTe338d66Sl3A?usp=sharing)
- ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_convnext-s: [notebook](https://colab.research.google.com/drive/1KnagfSmFJ69HfASLCfzCOCrZBZyL3y7-?usp=sharing)

### coordinate pretrained models

[this notebook](https://colab.research.google.com/drive/1TdmZCT86dFTiP2vd2uMPtNMaAsJKPkc2?usp=sharing) is pre-training code for coordinate prediction models with [this dataset](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/524500). model checkpoints are in [this dataset](https://www.kaggle.com/datasets/wadakoki/rsna-pretrained-models-for-coordinate/data)
