# 7th place solution -> 2-step MIL based strategy

Competition: mayo-clinic-strip-ai
Rank: #7
Source: https://www.kaggle.com/c/mayo-clinic-strip-ai/discussion/358130

It was a nice starting experience for me in this competition and big scoop of beginner's luck to go with. Nevertheless, learnt a lot and final methodology with all learning implemented worked best, which is satisfying. Thanks to all who shared hints, tips and suggestions during this competitions. Here is a summary of the methodology used:

</br>
## Preprocessing - Tile Creation and Selection

Tried a lot of things at the start but finally settled with tiles created from image downsized by a factor of 6 to balance signal loss and manageable number of tiles. Created and selected tiles by:

- Creating (224,224,3) shape tiles and applying simple threshold on image array unique values and their counts - for removing mostly monochrome and background tiles. 
- Reduced tile number further by running tiles through EfficientNetB0 (pretrained, ImageNet weights), scoring them and choosing top 50-60% tiles for each WSI (created another problem in tile number per image, addressed towards the end).
- Added augmented data for LAA class by creating extra tiles simply by padding image and shifting tile location in image by tile_size/2 in both directions. 

</br>
## MIL - Pseudo Labelled Tiles based Feature Extractor Training, Feature Aggregation and Classifier

Followed the tile feature embedding and then aggregation at slide level for further classification method. There are two main steps:

### 1. Feature Extractor Training
[This notebook](https://www.kaggle.com/code/icemantd/feature-cluster-tile-psuedo-label-train-mayo) has detail implementation of this part. The main points are:
- Trained EfficientnetB0 starting with ImageNet weights, first epoch used all tiles with slide level labels.
- Added random hue augmentation to tiles (details towards the end).
- Defined a feature extractor which is the whole network upto one layer (shape (1,1280)) before the last FC classification layer.
- Used tile features from extractor to apply cluster (kmeans) distance based pseudo labels to tiles for next iteration.

The logic used here is to attempt to provide the feature extractor the ability to not only give high output for the two classes but also a low output for out of class or non-critical tiles. Hence I used 'Other' label tiles with target [0.0] and LAA and CE with [1,0] and [0,1] respectively, with sigmoid activation and BCE loss. Every training epoch, features are created, clustering performed, and tiles scored on minimum distance from the negative class - i.e. top 20% highest min distance from LAA+Other feature clusters for CE tiles are labelled [0,1] and bottom 10% lowest min distance CE tiles are labelled [0,0] - and vice-versa for LAA tile pseudo labelling. The hope was this discriminates between the two classes, as well as between critical and non-critical tiles. The main clustering idea is inspired by [this](https://arxiv.org/pdf/2206.08861.pdf) paper.

- Used random sampling of CE class to balance classes and used a simple metric for training evaluation at slide level - maxpooling AUC scores, i.e. taking AUC scores for maximum predictions of both classes by any tiles in the slide. 4-fold CV AUC scores of 0.6-0.65.

### 2. Feature Aggregation, Classification and Inference
Obtained tile level features from feature extractor and adapted [this](https://arxiv.org/pdf/2011.08939.pdf) paper's methodology for feature aggregation at slide level, based on attention to tile with max output logit. 

- Built a simple random forest classifier using aggregated features at slide level and max predictions at slide level (any tile) for both classes as extra features.
- Used only top 25 dark tiles from any image for training classifier and for inference.
- Inference: Take top 25 dark tiles from image upto 2.2 Gb in size, create features from extractor and max logits from classifier head, aggregate and predict classifier probabilities ([notebook](https://www.kaggle.com/code/icemantd/new-predict-mil-logit-features-mayo-strip-ai/notebook))

</br>
## Some important observations
- Found out that tile selection was very important, especially if you are using smaller tiles. Observed that at one point, larger images were giving better log_loss scores but there are more smaller images to infer (at least in training dataset). So adjused the top% of tiles selected from larger images, especially because LAA class had large images and some images dominated the tile numbers and resulted in overfitting. This definitely helped in controlling penalty on bad log_loss over large number of images. Example of log loss spread over training data vs number of tiles per slide image (number of tiles per image represents original number created and represents size of image and not number of tiles selected for training):



- Observed from [this](https://arxiv.org/pdf/1902.06543.pdf) paper that adjusting hue of the image, even a little bit, resulted in better chance of predicting unseen strains better. So I implemented a random hue augmentation based on stretching/compressing the hue channel of each tile randomly by a factor betweem (0.85, 1.15). For example

[![]

- Due to the 2-step complexity, I could not come up with a robust end-to-end CV pipeline. Only spent time on feature extractor CV and evaluation using various data splits (60-40, 90-10) but ran into various issues with validation metric robustness. This part was the weakest for me and that's why I consider myself lucky to still land on something that kind of worked.

Hopefully some of you can point out where things could have been better or what was unecessary and even incorrectly implemented. Thanks.
