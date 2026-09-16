# 3d  place solution

Competition: hubmap-organ-segmentation
Rank: #3
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/354683

*First, we would like to thank the Armed Forces of Ukraine, Security Service of Ukraine, Defence Intelligence of Ukraine, State Emergency Service of Ukraine for providing safety and security to participate in this great competition, complete this work, and help science, technology, and business not stop and move forward.*

Also thanks to my teammates (@sakvaua , @igorkrashenyi, @alexkirnas), Kaggle team and competition organizers for this year HuBMAP + HPA event. It was a great pleasure to compete in order to bring benefit in biomedical sphere!
According to hosts the main challenge was:
```
Adapting models to function properly when presented with data that was prepared using a different protocol will be one of the core challenges of this competition. While this is expected to make the problem more difficult, developing models that generalize is a key goal of this endeavor.
```   
And it was really a key point to Hacking the Human Body this year :) So here is our approach 
# Data  
- Drop some completely (to our point of view) mislabeled lungs. IDS. Later the organizers clarified that those ids are not mislabeled but rather represent a different way to section alveoli. Anyway there were too few of them to be useful and they confused our models without providing any meaningful new data. So we decided to focus on the horizontally sectioned alveoli that look like bubbles.
```
12476, 127, 13189, 15124, 16564, 23252, 25516, 25945, 29610,30084,30500, 31139,31571, 7359, 8151
```
Lately we have pseudo-labeled it and added again to training 
- One of key points was adapting to wildly varying pixel sizes. The images scales ranged from 6.3um for prostate and down to 0.2um/pixel for large intestine. We tackled this issue by rescaling our train dataset to the target HuBMAP resolution. Though to increase the model’s receptive field we applied additional downscalers for larger images and upscalers for the prostate. 
We used one dataset rescaled to HuBMAP scales and another dataset with the original HPA scales. The latter one was not only important for HPA predictions (absent in the private LB) but also to provide some additional scaling information to the model. 
Here is our additional rescaling dict (original scale / additional scale). 

```
"prostate": 0.15 * 2,
"spleen": 1 * 2,
"lung": 0.5 * 2,
"kidney": 1 * 2,
"largeintestine": 1 * 2,
```
- One important aug was CutMix. We have used pretty aggressive CutMix ("prob": 0.5, "alpha": 1.0)  though the trick here was to apply CutMix only within the single organ class.
- We have trained CNN based models on 512 crops and segformer on 1024 crop. As for segformer bigger training crop it was pretty important, because results on 512 crop were much worse. While increasing training crop size for CNN - increased local validation but decreased HubMap LB score. Also we have sampled Non Empty Mask with probability 0.5
- Also we have used pretty heavy augs - Geometric, Color, Distortions, Scales. The overall pipeline is really long :). And it might be the key point to model performance and stability on. The main idea behind the color augmentation was to suggest the model that the color is not important and it had to look for the other clues
- To deal with the color shift between the DAB and H&E stains we used Histogram Matching of training pictures to H&E stained GTEX and HubMap images. We have used Historgram Matchin for all the images.

## Pseudo/Additional Data
We used additional data from GTEX and HPA portals to complement the initial training data. The GTEX data was especially important here because it was stained, similar to HuBMAP slides, with H&E. From GTEX we downloaded prostate, large intestine, kidneys and spleen data for patients with no apparent pathologies. We ignored lungs from GTEX as we couldn’t figure out how to segment those (and neither did our model). We were progressively adding GTEX images to our pipeline ending up with around 140 at the end of the competition. And pseudo labeled them with the same ensemble, as HPA
From the HPA site we used a plethora of DAB stained slide very similar to those provided by organizers. Overall, we have added between 57-61K of additional HPA images for each organ. 
We pseudo labeled all the external data with our ensamples, which scored 0.59 on HubMap and 0.81 on HPA+HubMap LB.
For selecting particular pseudo images for labeling we were inspired by [Semi-Supervised Segmentation of Salt Bodies in Seismic Images using an Ensemble of Convolutional Neural Networks](https://arxiv.org/pdf/1904.04445.pdf%3C/p%3E). We did not select best labels but just randomly sample the same dataset as training one from HPA pseudo and all GTEX dataset for training.
We have repeated pseudo labeling twice 

## Composed training dataset
All our training dataset was composed from 3 subsets:
1. HPA train fold + HPA pseudo with Pixel Size rescale and Hist Matching - repeated X8
2. HPA train fold + HPA pseudo - repeated X4
3. GTEX pseudo with Pixel Size rescale - repeated X16
# Models
We have used [SMP](https://github.com/qubvel/segmentation_models.pytorch). And final ensemble :
- Unet ++. Encoder Effnet B7 (noisy student pretrain). First HPA pseudo iteration. X5 - folds
- Unet. Encoder Mit B5 (imagenet pretrain). Second HPA pseudo iteration. X5 - folds
- Unet . Encoder Effnet B7 (noisy student pretrain) + Point Rand only for training. First HPA pseudo iteration. X5 - folds  
- Unet . Encoder Effnet B7 (noisy student pretrain) + Point Rand only for training. Second HPA pseudo iteration. X5 - folds  
Public Hubmap: 0.61453
Public Hubmap+HPA: 0.83983
Private: 0.83266
Local OOF: 0.8493729371691843
Local organ scores:
kidney: 0.954695
largeintestine: 0.913645
lung: 0.488490
prostate: 0.833822
spleen: 0.834292
As for [PointRand](https://arxiv.org/abs/1912.08193) we have used it only as additional loss as regularization while training 

Interesting Point is that CNNs showed better performance on public HubMap ~ 0.61 comparing to Segformers ~ 0.60, while Segformers outperformed CNNs on local validation ~0.85-0.86, while CNNs ~0.83-0.84. BUT on Private Segformers outperformed CNNs and our best Private submit (not selected) includes:
- Unet ++. Encoder Effnet B7 (noisy student pretrain). First HPA pseudo iteration. X5 - folds
- Unet. Encoder Mit B5 (imagenet pretrain). Second HPA pseudo iteration. X5 - folds
- Unet. Encoder Mit B3 (imagenet pretrain). First HPA pseudo iteration. X5 - folds
Public Hubmap: 0.60931
Public Hubmap+HPA: ???
Private: 0.83419
Local OOF: 0.8542834156960896
Local organ scores:
kidney: 0.956776
largeintestine: 0.917554
lung: 0.488331
prostate: 0.838744
spleen: 0.848721
It may granted us second place :) 

# Training
- We used one channel output
- Adam 1e-3
- ReduceLROnPlateau : patience=3; factor=0.5; min_lr=1e-7 by valid dice
- losses: softbce + tversky + focal + jaccard + (for some models) point rand loss * 2
- fp16

# Validation 
- 5 folds CV with stratification by organs 
- Predict image by image (batch size 1) in full scale with rescale to HubMap scale

# Inference
- Average 3 best checkpoints for each fold - just average model weights
- Mean Fold prediction
- 4 Flips TTA
- Remove small regions - compute relative area on HPA data and take as threshold - <0.5 quantile
- Predict image in full scale. We have only used sliding window for Segformers (1024 and 0.75 overlap) just because of cuda out of memory problem

**Inference Kernel** : https://www.kaggle.com/code/vladimirsydor/hubmap-2021-inference-v1/notebook?scriptVersionId=106283407
**GitHub** : https://github.com/VSydorskyy/hubmap_2022_htt_solution 
**Paper** : https://arxiv.org/abs/2305.02148
