# 43rd solution: using the MedNext fork WMCSFB

Competition: czii-cryo-et-object-identification
Rank: #43
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561957

# my CZII CryoET 43rd solution

This is not the winning solution but I drop here my efforts in case anybody is interested.

I start by highlighting that pratically all the training of models and most of the coding was done using Kaggle computing, as my home computer was not sufficient to explore further.

## Choosing model: WMCSFB:

very long story...

- I started with a custom version of LeopardGecko, which is for volumetric segmentation and uses 2D unets and a multi-planar approach. This is based in another similar package called volume-segmantics. It worked very poorly.

- wollny 3D unet. Very poor results

- MedNext. https://github.com/MIC-DKFZ/MedNeXt, https://arxiv.org/abs/2303.09975. Very good results, so I for a while I settled with this model and used to improve training algorithm.

- It then improved further using a Mednext-like model called WMCSFB, by Wenzhao Zhao. See ref https://arxiv.org/abs/2402.16825, and github https://github.com/ZhaoWenzhao/WMCSFB. This was based in the first version of Mednext which did not have the support for GRN, I modified to implement this grn functionality as in Mednext version v2. An interesting feature of this model is the use of spherical Fourier-Bessel functions in initializing the first layers of the convnext/mednext model. The model code was all included in the training notebook.

The code used for training is provided here: https://www.kaggle.com/code/perdigao1/wmcsfb-grn-train?scriptVersionId=220076028

The model I used was the small 'S'. Bigger versions of the model did not result in better results.



# Strategy for training

- copick software was NOT USED. I implemented the creation of masks myself.
- masks generated using 1/3 or the particle radius
- 5 of the volumes used for training initially.
- One important thing I noticed was that if volumes 5_4 and 69_2 were used during training, it seemed to have detrimental effect in the final score(s), suggesting there is something wrong with the data or masks in these volumes. Many good scoring notebooks also seem to exclude these volumes from training. I didn't really find a good explanation for this due to limited compute availability.
- Training volumes: 64x64x64 size, with the particle in the centre (unless is an edge particle)
- Added random volumes to the training data, so that 2/3 of the data would be random volumes
- simple normalize per 64x64x64 volume, using mean and stddev.
- Augmentations: one voxel shift, contrast + brightness, 90deg rotation on the XY plane, flips along all axis, voxel swap (custom coded), gauss noise
- Loss function: Custom coded Tverski loss that support class weights (0.25, 1.0, 1.0, 1.0, 4.0, 4.0) respectively for ( background, 'ribosome', 'virus-like-particle', 'apo-ferritin', 'beta-galactosidase', 'thyroglobulin' )  (note the "non-scoring" particle is missing)
- Metric, IoU (smp.utils.metrics.IoU())
- Data is split into train and valid using sklearn's StratifiedKFold (4 folds).
- Optimizer: AdamW, Gradscaler
- Train using onecycleLR scheduler
- Number of epochs and further results are presented in the CFG variable defined at the start of the notebook

Other additional training schedulers followed and there was a mechanism to pick the "best" training state by the metric score. However, the best result was obtained from this first step, so there is no point to elaborate further here.

# Inference

https://www.kaggle.com/code/perdigao1/wmcsfb-grn-subm-inference

For inference, the volumes were patched (blocks) to volumes of (160,160,160) and overlap (14,14,14). Only the middle parts were considered except at edges. The result (probabilities) was then argmaxed, and connected components was run to get the centroids. The list of centroids was then parsed through DBSCAN that gave marginally better results. The DBSCAN idea came from some of the notebooks that were shared.

The results I obtained were quite good visually. With some particles it appeared that the prediction was better quality than the ground truth itself, in terms of being at the correct centre, or sometimes it appeared that there were missing annotations. As such I felt it was difficult to come up with a better way to improve

Something that I started exploring but did not complete were 3D Unets with complex numbers (phase), and quaternions. There are some examples in literature. Some mathematical intuition tells me that these unets **should** work better as the fully trained model may converge to something that resembles a fourier or bessel kernels, as in the solution used here. However there needs to be significant changes in the way the normalisation, residuals and activations are applied. I tried some draft versions but these either did not train well or gave poor result, public score max of 0.5 which is not a bad start.
