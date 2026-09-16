# 20th Place Solution - UBC-OCEAN

Competition: UBC-OCEAN
Rank: #20
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465356

Although I am posting the write-up, this was a great team effort by @kevin0912 and me. Also, thanks to UBC for hosting this competition, it was a fun competition, and interesting working with such large images! 

Our solution is based on a multiple instance learning (MIL) architecture with attention pooling. We use an ensemble of `efficientnet_b2`, `tf_efficientnetv2_b2.in1k` and `regnety_016.tv2_in1k` backbones trained on sequences of 8 x 1280 x 1280 images, and ignore the `other` class. We also apply light TTA during inference (rot90, flips, transpose, random image order).

[Cropper]

## Strategies

**Efficient Tiling**

We select tiles from WSIs based on the darkest median pixel value. To make the pipeline more efficient, we use multiprocessing on 3 CPU cores, and prefilter crop locations using the smaller thumbnail images. This prefiltering selects the largest area of tissue on the slide and ignores other smaller areas of tissue.

For TMAs, we take 5 central crops of size 2560 x 2560 and resize to 1280 x 1280 to match WSI magnification. 

[Cropper]

Although efficient, a limitation of the pipeline is that it may not extract informative tiles from each image. We also experimented with a lightweight tile classifier trained on the ~150 segmentation masks, but this did not improve tile selection.

**Modeling**

We trained each model for 20-30 epochs with heavy augmentations and SWA (Stochastic Weight Averaging). Most models were trained on all the WSIs and TMAs, but some were trained using synthetically generated TMAs (aka. TMA Planets) from the [supplemental masks](https://www.kaggle.com/datasets/sohier/ubc-ovarian-cancer-competition-supplemental-masks). We would likely have explored TMA planets further but we were skeptical of the mask quality, and low count relative to the total number of WSIs.

[Cropper]

**OOF Relabel + Remove**

Based on [Noli Alonso's comments](https://www.kaggle.com/competitions/UBC-OCEAN/discussion/445804#2559062), we removed ~5% of the images and relabelled 8 images. We used a similar denoising method to that in the [1st place solution](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/discussion/169143) of the [PANDA Competition](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/overview).

```
relabel_dict = {
    '15583': 'MC',
    '51215': 'LGSC', 
    '21432': 'CC',
    '50878': 'LGSC',
    '19569': 'MC',
    '38097': 'EC',
    '29084': 'CC',
    '63836': 'LGSC',
}
```

## External Data

The only external dataset we used was the [Ovarian Carcinoma Histopathology Dataset (SFU)](https://www.medicalimageanalysis.com/data/ovarian-carcinomas-histopathology-dataset). This dataset had 80 WSIs at 40x magnification from 6 different pathology centers.

Class distribution: `{'HGSC': 30, 'CC': 20, 'EC': 11, 'MC': 10, 'LGSC': 9}`

## Did not work for us

- Larger backbones
- Lightweight tile classifier
- Stain normalization (staintools, stainnet, etc.)
- JPGs

## Frameworks

- [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) (training)
- [Weights + Biases](https://wandb.ai/site) (logging)
- [Timm](https://huggingface.co/timm) (backbones)
