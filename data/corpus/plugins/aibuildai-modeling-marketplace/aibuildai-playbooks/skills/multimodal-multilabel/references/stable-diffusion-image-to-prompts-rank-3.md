# 3rd place solution

Competition: stable-diffusion-image-to-prompts
Rank: #3
Source: https://www.kaggle.com/c/stable-diffusion-image-to-prompts/discussion/410686

Thanks for organizers hosting a nice competition.

My approach is based on CLIP model with directly predicting 384 embedding vector following [this baseline notebook](https://www.kaggle.com/code/shoheiazuma/stable-diffusion-vit-baseline-train).

## Dataset
I used about ~400K data. The validation score of Diffusion DB was well correlated to LB score but still 0.025 ~ 0.03 gap.

* [Vizwiz image caption](https://vizwiz.org/tasks-and-dathasets/image-captioning/) ~70k
  - This was key dataset for my experiment.
    + Training with this data, local val vs LB score -> 0.5415/0.5309, but without this 0.5528/0.48765
    + I think this caption dataset is generally more descriptive, diverse and longer than COCO.
    + Original training samples are 23K at this dataset, but each training sample has max 5 captions, so sampled 3 captions for each training sample.

* Diffusion DB, 300k
  - only using SD2 images, which I personally generated for improving local val/LB correlation
  - 210K samples from prompt filtering and another 80k images from hard sampling using a trained model.

* COCO, 25k
  - Under training with Vizwiz, this dataset contribution was relatively low.

* [Lexica.art](https://www.kaggle.com/datasets/motono0223/gustavosta-stable-diffusion-prompts-sd2-v2), 10k

## Model
* only using CLIP models
    - My baseline model was ViT Base 224(laion2B), almost all experiments were done on this model.  Final score for this model was, local val/Public/Private=0.6402/0.61402/0.61357
    - Single best model. ViT L 336(open ai), local val/Public/Private=0.6658/0.63557/0.63425
        - I could not find good training parameters for larger models, ViT-H and ConvNext xxLarge.
    - Best submission, ensemble of ViT-L(laion2b, openai, datacompxl), ViT-H, and ConvNext Large/xxLarge,  local val/Public/Private=0.6791/0.64935/0.64814

## Training
### CLIP fine tuning
Fine tuning CLIP model needs extra care compared to imagenet weight, we have to keep original CLIP weight as possible as we can to get best performance.
I found the following two methods improve score by ~ +0.02 in total and used a lot of time to find best hyperparameters.

1. [LP-FT(Linear Probe and then Fine Tune)](https://arxiv.org/pdf/2202.10054.pdf)
    - This was studied at [the top-1 solution  of previous competition](https://arxiv.org/ftp/arxiv/papers/2210/2210.08473.pdf ) and also other teams at this competition.

2. Combination of EMA and layer-wise learning rate decay
    - This has been studied at [this CLIP fine tuning paper](https://arxiv.org/abs/2212.06138).
    - Best parameter for ViT-B was (EMA decay, layer-wise decay) = (0.9998, 0.4)
    - But this did not work for ViT-L, layer-wise decay = 0.6 was best. It seemed decay factor for initial layer should be around 1e-6 so we have to choose large decay late for larger models.

These two methods has not only direct effect but also indirect effect, because by these methods we can increase learning rate with keeping CLIP original weight and get better result.

### Augmentation
* Crop/RandomErase/RandAug(without posterize, solarize and equalize)
    - This contributed to local val score, ~ +0.007 and this enabled longer training.
    - Strong image transformation did not fit this competition, I tried to find weak one.
    - Horizontal Flip worsen performance

* Same prompt and different seed
    - 10% of Diffusion DB and 20% of Vizwiz samples have 3 images which are belong to the same prompt but different SD2 generation seed. For  each training epoch one of 3 images randomly chosen.
    - This also contributed to the score and validation loss curve.

* [Invisible watermark](https://github.com/ShieldMnt/invisible-watermark) augmentation
    - Adding watermark on input images during training with the following [stability ai repository](https://github.com/Stability-AI/stablediffusion/blob/334969331438036c1b4fb529c262f78499870d86/scripts/txt2img.py#L363)
    - This did not contribute much to the score, only ~ +2.0e-5

### Code
This is [my submission notebook](https://www.kaggle.com/code/sai11fkaneko/3rd-place-solution/notebook).
