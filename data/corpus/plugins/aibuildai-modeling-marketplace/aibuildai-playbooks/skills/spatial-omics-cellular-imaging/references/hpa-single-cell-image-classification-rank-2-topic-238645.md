# HPA 2nd Place Solution [red.ai]

Competition: hpa-single-cell-image-classification
Rank: #2
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238645

## Preface
As promised, we will share our detailed solution within 24 hours. We would like to thank the organizers for this awesome competition since all of us had no experience in dealing with weakly-supervised classification problems and we have learned a lot from the the kind sharings by other kagglers and self-discoveries. The organizers are very active in this competition; huge props to all of you. I am also grateful for my teammates for making my journey to **Kaggle Competition Grandmaster** smooth and gratifying. To say I am excited is a huge under-statement. Without further ado, let's dive into our solution.

## TLDR
Our solution consists of a total of 3 simple pipelines. We did not use any advanced techniques from any paper but we tried to understand the data well and build our model architecture w.r.t the problem statement. Here is a diagram for our final pipeline:



## Pipeline 1: Duo-Branch Cell Model
- Motivation: A duo-branch(head) cell model was designed in a way that it takes cell tiles as input but has the ability to predict both as cell-level and image-level. Multi-tasking has been shown to be effective in improving model learning. A strong champion in dota2 called Jakiro also has two heads.
- Loss formulation: since the output is cell-level and image-level, we need two losses for both outputs. The final loss is the weighted sum of cell-level loss and image-level loss. We used basic **BCE** loss for both cell-level and image-level. For cell-level, the labels are not certain so it's intuitive to assign a lower weight (=0.1). l = 0.1*loss_cell + loss_image
- Data: we used original data, external data shared by Phil as well as some rare class samples by using the API. The input size for a single cell is 256.
- Training details: we take 4-channel images and crop&resize the cells first; then we random sample N (=16) cells as input of our network. The cells are flattened as a large batch then we feed them into a CNN and backprop. For data augmentations we used dihedral, shift, rotate, scale, distortions, brightness contrast and cutout. The heavy data augmentations allows the model to better generalize as cells can be in any forms in reality. We train 5 folds and 20 epochs each. A single Model (b3, 256) takes about 30 hours to train on a single RTX3090.
- Inference: for each image, the pooled features are concatenated and feed into the last linear layer to predict at a cell-level. We generate image level prediction and cell level prediction and calculate their product as our final prediction.
- Result: with 16xTTA(scale, rotate, flip at random), 256 size cell-tiles, N=16, ensemble of efficientnet B3, B5 resnet200d and se_resnext50 backbone, our model score `0.550` on the public leaderboard and `0.550` on the private leaderboard. This single architecture can achieve second place in this competition.



## Pipeline 2: Image-level Model

- **Misc:** The image level model is similar to last HPA's competition. We feed in whole images and perform a multi-label classification. It may surprise you, this pipeline is developed in fastai. We have found fastai's many implementations to be extremely slow (for instance, resize) and had gone through many days of debugging during the final inference phase; we spent a whole last week attempting to figure out how to submit. In the end, we optimized the fast.ai inference code by a lot that helped us cut the inference time almost twice. Luckily, hard work paid off.
- **Motivation:** we can train at an image level but predict at a cell-level (with other cells masked) and the result is very promising. We decide to add this our pipeline.
- **Data:** we used all data available on HPA official website and resized it to 512 using only RGB 3 channels.
- **Training details:** we train 20 epochs with class weight [0.1, 1., 0.5, 1., 1., 1., 1., 0.5, 1., 1., 1., 10., 1., 0.5, 0.5, 5, 0.2, 0.5, 1.] and BCE loss for 2 folds only. We used average precision score  for checkpointing. For data augmentations, we used fastai's `aug_transforms(flip_vert=True, max_lighting=0.1, max_warp=0.1, p_affine=0.5, p_lighting=0.5)`
- **Result:** we had 10 (5x2folds) models and we took the mean of the final output. And we use them to predict at both cell-level and image-level. We take the mean as our final output.



## Pipeline 3: Cell-level Model
- **Motivation:** we can train at cell-level using the image-level labels but it's a bit counter intuitive. Since his will introduce lots of noise as image-level labels are not ground truth for cells so we think it's beneficial to train less epochs. We ended up only training 2 epochs (1 with backbone freezed and 1 with backbone unfreezed).
- **Data:** we used all data available on HPA official website, use the cell segmentor to crop the cells and resized the cells to 168 using only RGB 3 channels. There are a total of 1620178 cropped cells.
- **Training details:** we used fastai's built-in `finetune` and fastai's learning rate finder to train only 2 epochs with the same class weight [0.1, 1., 0.5, 1., 1., 1., 1., 0.5, 1., 1., 1., 10., 1., 0.5, 0.5, 5, 0.2, 0.5, 1.] and bce loss. We did not use anything for validation.
- **Result:** we had 10 (10x1folds) models. We predicted at cell-level and simply took the mean of the final output.



## Segmentation Model

We are inspired by @samusram Even Faster HPA Cell Segmentation and @alexanderriedel Segmentation with a Scaling Factor, we modified the original HPA Segmentator to gain speed but keep the segmentation quality.
* Post-processing: We slightly changed label_cell function from the original implementation. We found that in many cases, border cells are segmented in a wrong way: some of them are combined together with border cells that have no nuclei (or it’s outside of the image). We tweaked the watershed distance threshold in order to separate cells masks a little bit further from each other than they were before, then we ignored the masks on the border that became separated from the main cell. Furthermore, we removed the border cells with nuclei whose area was less than a half of the median area of the non-border nuclei on the image. And we also removed the cells that did not have the corresponding nuclei. Below is an example of a difference between original label_cell implementation (left) and ours (right):





## Arcface Model
We also trained an arcface model with eca_nfnet_l0 backbone to classify antibody_id. Antibody_id can be found on the HPA's official website in a XML file. There are a total of 11582 antibody_id and it's extremely difficult to train. We used arc_margin_product and bce loss to train for 15 epochs then extracted the feature embeddings for the whole dataset. We used faiss_gpu library for cosine similarity search during inference. It worked well on the public leaderboard but it didn't quite work on the private leaderboard.

## Duplicate samples.
We found about ~400 image in public test set duplicated either within the train set or the external data. You can check the csv file at https://www.kaggle.com/steamedsheep/hpa-2021-duplicated-sample. Our public leaderboard score, excluding the boost from duplicates is about 0.58, we have a relative consistent gap w.r.t. the 1st place in both public and private leaderboard.

## Things that didn't work
- Segmentation post-processing on scaled-up outputs of the segmentator led to a slight decrease in the score
- Tiling a plot with a single cell and classifying such cells with the image level models.

## Solution code
* Pipeline 1's code is now available at [github](https://github.com/iseekwonderful/HPA-singlecell-2nd-dual-head-pipeline)
