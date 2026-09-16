# 13th Place Solution for the UBC-OCEAN Competition

Competition: UBC-OCEAN
Rank: #13
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465358

First of all we would like to thank Kaggle and the sponsors for this interesting research competition.


# Context



* Business context: [https://www.kaggle.com/competitions/UBC-OCEAN/overview](https://www.kaggle.com/competitions/UBC-OCEAN/overview)
* Data context: [https://www.kaggle.com/competitions/UBC-OCEAN/data](https://www.kaggle.com/competitions/UBC-OCEAN/data)


# Overview of the approach

Our final solution was a combination of separated high scoring TMA and WSI models. For WSI models we’ve used some self-supervised learning (SSL) pretrained features extractor executed on 224 pixels tiles followed by Multiple Instance Learning (MIL) models. Same for TMA model with additional regular Transformer and CNN backbones. Outliers detection takes place in postprocessing and relies on both embedding distance and probabilities distributions.


# Detail of the submission

**WSI models:**




**TMA models:**



**Post processing:**

TMA models have been trained with ArcFace loss and an ArcMarginProduct sub center module to try to separate embeddings space as much as possible per class. It allows detecting outliers based on a similarity distance. Our distance threshold has been fine tuned on public LB as we didn’t have any solid sample of real outlier/rare class. [Faiss](https://github.com/facebookresearch/faiss) package has been used to find nearest neighbors.

We also performed another thresholding based on probabilities distributions. When all probabilities are low enough we switch the predicted class as outlier. The threshold has been calibrated on CV to avoid more than 10% outliers and checked on public LB.


# WSI model training

Before training the WSI model, our medical intuition supported by this article<sup>1</sup> led us to hypothesize that the relevant information for the subtype prediction was probably more located at low level. All the 513 available WSI in the training set were thus downscaled to 10x magnification. Since all these WSI had a black unicolor background, we then performed an otsu thresholding, in order to discard all the background tiles. We then tiled all the detected tissue into N non-overlapping 224px tiles. Insofar as it was not possible to infer tumor segmentation during submission because of the time limitation, we decided to keep all the tumor and the non-tumor tiles for training.

All these N tiles were then encoded with CTransPath and Lunit-DINO, 2 features extractors trained using self-supervised learning on diverse pathology dataset. According to this article<sup>2</sup> about the robustness of these models to stain variations, we did not perform any kind of augmentation or normalization preprocessing.

We then trained and evaluated several MIL architecture into a weighted ensemble. The best CV were obtained by combining three of them : 



* Clustering-constrained attention MIL ([https://github.com/mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM))
* Dual-stream MIL ([https://github.com/binli123/dsmil-wsi](https://github.com/binli123/dsmil-wsi))
* A weighted sum of the embeddings.

MIL training procedure and parameters:



* CV4, Stratified Group KFold
* No augmentation
* No normalization
* Batch size =1, epochs = 32
* AdamW optimizer, CosineAnnealingLR, LR=5e-3
* Cross-Entropy Loss


# TMA model training

The UBC training dataset was coming with only 25 TMA samples and we know, according to the description, that TMA in the test set are the majority. We’ve detected around 65% to 70% of images with sides less than 6000 pixels. We’ve decided to generate some TMA based on the WSI provided in the training set. We’ve developed a custom augmentation that is detailed in this notebook: [https://www.kaggle.com/code/mpware/ubc-tma-generator-from-wsi](https://www.kaggle.com/code/mpware/ubc-tma-generator-from-wsi)

```python
def tma_augmentation(p=1.0):
    return A.Compose([
        SimulateTMA((-1, -1), radius_ratio=(0.6, 1.0), ellipse_ratio=(0.85, 1.15), angle=(-90., 90.), 
                    background_color=(-1, -1, -1), background_color_ratio=(0.80, 1.0), 
                    noise_level=(20./5, 100./5), black_replacement_color=None, p=1.0, always_apply=True),
        A.OneOf([
            Stainer(ref_images=tma_images, method='vahadane', luminosity=True, p=0.34),
            Stainer(ref_images=None, method='macenko', luminosity=False, p=0.33),
            Stainer(ref_images=tma_images, method='reinhard', luminosity=False, p=0.33),
        ], p=0.60),        
    ], p=p)
```

The idea is to identify tiles with tumoral tissue and crop an ellipse shape as could be a real TMA picked by an operator.




The crops are then augmented with stains based on the 25 TMA as references. As the WSI magnification is mainly x20 the generated TMA are also x20. Here are some generated samples:



A final step was to review them manually to drop bad generated TMA (especially when tumor mask was not available / complete). Our best private LB (0.58) was with such validated TMA. Unfortunately we did not select it as final submission.

**MIL** training procedure and parameters:


* CV4, Stratified Group KFold
* Random batch sampler to balance samples
* Augmentations: Stain: Vahadane, Macenko, Reinhard
* Mask on attention, batch size = 32, epochs = 32
* EMA
* AdamW optimizer, CosineAnnealingLR, LR=1e-3
* Cross Entropy Loss

<br/>
**Transformer/CNN** training procedure and parameters:


* ImageNet pretrained backbones ([Timm](https://github.com/huggingface/pytorch-image-models)):
    * tiny_vit_21m_512.dist_in22k_ft_in1k
    * tf_efficientnetv2_s_in21ft1k
* Augmentations: 
    * H/V flips, Rot90
    * Stain: Vahadane, Macenko, Reinhard
    * Random BrightnessContrast/Gamma, HueSaturationValue, ColorJitter, CLAHE
    * GaussianBlur, MotionBlur, GaussNoise
    * Cut Mix, DropOut
* EMA
* Batch size = 32, Epochs = 32
* AdamW optimizer, CosineAnnealingLR, LR=1e-4
* Cross Entropy Loss

Models have been trained with full data after checking stability on cross validation.

Here is a 2D t-SNE projection of TinyVit trained embeddings on generated 23k TMA:




# Other useful strategies or approaches

Validation was quite difficult, MIL models were overfitting quite fast. Using EMA helped to limit it. 


# Model inference

Most of the inference time was lost in image loading/tiling. We’ve implemented a multiprocess inference to benefit from all CPUs but optimized to balance the memory issues due to concurrent large images loading. It reduced the loading + tiling of all images to around 5h30-6h. Features exaction was the most time consuming task that is why we’ve limited ourselves to the two best ones. We’ve limited the number of tiles to 350 max and at the end our inference ran in around 11h15-30min.


# What did not work or improve?

A quick sum up of what did not work or not improve:

Resnet50-based features extractors such as RetCCL and Lunit-BT.

External data:



* [TCGA](https://portal.gdc.cancer.gov/repository?facetTab=cases&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.primary_site%22%2C%22value%22%3A%5B%22ovary%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Slide%20Image%22%5D%7D%7D%5D%7D)
* [ATEC](https://www.cancerimagingarchive.net/collection/ovarian-bevacizumab-response/)

<br/>
Usually adding more data is always better but here it did not help on both CV and LB. However the quality of many slides was very bad and could explain it. Also the labeling of some was not obvious.

Train a WSI model based on ImageNet pre-trained backbone. It worked but Ctranspath and LunitDINO outperformed it.

Down scale to x5 for WSI models (instead of x10)

Train a tumor segmentation model in order to sample tumor TMA, but since subtypes have significant morphological variations, we preferred to train a stroma segmentation model and predict the carcinoma mask by complementarity. Finally it was impossible to set up a unique threshold for TMA selection because of high variation in epithelial surface area between solid and mucinous architecture. 

Pseudo labeling has not been tried.


# Sources


* 1) Deep Learning for Detecting BRCA Mutations in High-Grade Ovarian Cancer Based on an Innovative Tumor Segmentation Method From Whole Slide Images: [https://www.modernpathology.org/article/S0893-3952(23)00209-0/fulltext](https://www.modernpathology.org/article/S0893-3952(23)00209-0/fulltext)
* 2) A Good Feature Extractor Is All You Need for Weakly Supervised Learning in Histopathology: [https://arxiv.org/pdf/2311.11772.pdf](https://arxiv.org/pdf/2311.11772.pdf)

<br/>
We had a lot of fun solving this kaggle. It was a lot of data to handle, in addition to the ML challenge it was an optimization challenge to make the inference fast.

<br/>
Raphaël Bourgade and MPWARE
