# 27th Place solution: Algorithm vs Memmory or Transformers is all you need, 0.39/0.44

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #27
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539494

Thank you so much to everyone, especially to organizers and people who shared insights. Has been a long and stressful but enjoyable experience. There is a lot to comment and explain, but at the moment  a summary accross the full process.

**Starting point:**

https://www.kaggle.com/code/abhinavsuri/anatomy-image-visualization-overview-rsna-raids

This notebook introduced me perfectly to the problem and gave me the fundamentals of my approach since first contact. Often professionals base their diagnosis on 2D Sagittal T1 analysis for foraminal, 2D Axial T2 analysis for subarticular and 2D Sagittal T2 and Axial T2 analysis for spinal.

The main problem is that slices doesn't match perfectly between them, at least without metadata. So I've decided to use a flexible architecture to handle them, Transformers.

**2D UNet for ROI localization in images:**

https://www.kaggle.com/code/sacuscreed/sagittal-t1-sagittal-level-segmentation
https://www.kaggle.com/code/sacuscreed/sagittal-t2-sagittal-level-segmentation
https://www.kaggle.com/code/sacuscreed/axial-t2-axial-side-segmentation

[Predicted and True levels]

[Predicted and True levels]

[Predicted and True sides]

Those coordinates together allowed to point backbone slices in Sagittal T1, spine slices in Sagittal T2 and to assign levels to Axial T2 slices. Special thanks to @hengck23 for [2D to 3D proejection for DICOM](https://www.kaggle.com/code/hengck23/2d-to-3d-projection-for-dicom)



Is important to mention that I've only used competition data. By data processing I've been able to impute coordinates to slices that hasn't been labeled. All slices between left and right labels for forminal, all neighbor slices in a range of D//5 for spinal and neighbor slices for subarticular. Increasing considerably the amount of data available. I've also duplicated Axial T2 slices by flipping images and coordinates properly.

**CNN for ROI localization in crops:**

At first, I started by feeding Transformer "full-slice sandwiches" of the corresponding level or side.

@sergiosaharovskiy "so it means for your pipeline you do not find the centroid, but rather taking the entire axial slice, which gives 0.47 LB overall :D?" That's exactly what I did.

But because the above comment and the unsolved problem of effectively downsample big MRIs, with the consequent possible valuable information loss, I've started to try to find ROI also in crop slices.

https://www.kaggle.com/code/sacuscreed/sagittal-t1-foramina-discriminator
https://www.kaggle.com/code/sacuscreed/sagittal-t2-spine-discriminator



Again naive but reasonable label imputation was a key. Explicitely labeled crops were directly trusted. Immediate neighbors were excluded. And remaining crops were labeled as negatives.

At first the idea was to directly select the crops of interest of each "sandwich" but results were inconsistent so finally I've decided to aproximate ROI with DICOM correspondence between MRIs and use this discriminators as starting encoders for Transformer. Which resulted in considerably smoother trainings.

**DICOM for axial level assignation:**

https://www.kaggle.com/code/sacuscreed/getting-true-axial-levels



A direct implementation from [[ver.1] demo workflow: 2-stage approach](url). I've been working in a less literal adaptation. But time ran out and this one worked perfectly. Axial T2 slices are assigned to levels as the closer planes to the respective level coordinates in middle Sagittal slice.

**ViT over crops for final predictions:**

https://www.kaggle.com/code/sacuscreed/sagittal-t1-foraminal-prediction
https://www.kaggle.com/code/sacuscreed/sagittal-t2-spinal-prediction
https://www.kaggle.com/code/sacuscreed/axial-t2-subarticular-prediction
https://www.kaggle.com/code/sacuscreed/axial-t2-spinal-prediction-1-4
https://www.kaggle.com/code/sacuscreed/axial-t2-spinal-prediction-5

Finally I've trained five fold CV ViTs feeding Transformer with different sequences of crops using ResNet18 as encoder. Sagittal T1 crops for foraminal predictions:

[Sagittal T1 crops]

Sagittal T2 crops for one source of spinal predictions:

[Sagittal T2 crops]

Axial T2 crops for subarticular predictions:

[Axial T2 subarticular crops]

And Axial T2 crops for second source of spinal predictions:

[Axial T2 spinal crops]

I've been working cyclically between pathologies trying to transfer what I have learned each time to the next one. Trying to unify architectures. The main structure is a Transformer for crops in each level/side followed by a second Transformer between levels and sides. I've been experimenting with different positional encoding:

1) Learnable or not.
 
2) Distinguishing sides or not.

3) Absolute or relative to a Lmax as normalized position.

I would say that best approach is to absolute encode crops, with not learnable positional encodings and without distinguish sides. That is grouping second Transformer in sequence of 10 (2 for each level) since sides should be equivalent. The diagnosis shouldn't change depending on what side. I've updated this schema for Sagittal T1 and T2 predictions. But I haven't been able to traslate it to Axial T2 ones. In this case although original trainings were very unstable, they achieved at the end better performance. So I kept them.

**Inference:**

https://www.kaggle.com/code/sacuscreed/rsna-private-submission-v2

**Old code:**

https://www.kaggle.com/code/sacuscreed/old-axial-t2-axial-side-segmentation
https://www.kaggle.com/code/sacuscreed/old-sagittal-t2-segmentation
https://www.kaggle.com/code/sacuscreed/sagittal-t1-direction-training
https://www.kaggle.com/code/sacuscreed/sagittal-t1-axial-side-segmentation
https://www.kaggle.com/code/sacuscreed/sagittal-t2-axial-spine-segmentation
