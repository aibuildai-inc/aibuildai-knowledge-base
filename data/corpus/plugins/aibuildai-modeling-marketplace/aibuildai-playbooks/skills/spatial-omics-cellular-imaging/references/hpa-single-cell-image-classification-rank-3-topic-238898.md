# 3rd place Dieter part

Competition: hpa-single-cell-image-classification
Rank: #3
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238898

Thanks to Kaggle and hosts for this very interesting competition with a tricky setup. Also huge props to my team mates @zfturbo and @mpware who carried me to 3rd place. Since we worked quite independently on the models we thought it helps readability to split the work in 3 posts.  In the following, I want to give a rough overview of my part of our solution.

### TLDR
I did an ensemble of several ideas to tackle the weak label problem. The first model type works on cropped single cells which are weighted using trainable attention to derive a bag-of-cell prediction on which the loss wrt to the weak image label is calculated. The second model type is on image level. Here cell masks derived using HPA Cell segmentor are used to extract local features and derive a single cell feature vector. All single cell feature vectors are weighted to derive with an image level label. The third model type is an image level model which does not account for single cells, but helps the overall prediction ranking.

### Data setup & CV
I settled early on the hyperparameters for the HPA cell segmenter, which I slightly modified to run faster. I generated masks for all images and used those and fixed single cell ids through the competition. For cross-validation I identified clusters of similar/ duplicate images and used GroupKfold to create a robust validation scheme. I used Public HPA data and images from the first HPA competition, which were not included here.

### Models


#### Single Cell Model

I tried different ways how to create a bag-of-cell model to leverage the weak labels. The following worked very well and is quite similar to @wowfattie solution:

1. Use HPA Cell Segmenter to crop single cells
2. Feed batches x n_cells into the model
3. Use attention pooling to train weighting of cells within a bag and derive bag-prediction
4. BCE loss between image label and bag-predictiction

The following gives a brief illustration (e.g. bs =1):

[model1]

I want to note that the single cell model turned out to be especially robust on private LB as it can better handle high SCV.

#### Image model with aggregated local features
The second architecture uses the complete image but selects local features using a previously generated cell mask 

Cell mask is resized to 16x16 to map local feature vectors of backbone output (which has size (bs,1024,16,16)) to single cells and hence create an embedding vector for each cell. I weight the embedding vectors using trainable attention to derive the image level label. 

[model2]

I used backbones SE-ResNeXt26 and SE-ResNeXt101 from timm repository for these 2 model types.

The third model is an efficientnet-b7 trained on image level and predicts the same label for each cell within an image.


#### Compressing into one

I want to note that by generating single cell out-of-fold predictions using this ensemble and training a single cell model with that results in a good but lightweight single model (Public LB 0.524)


Thanks for reading. Happy to answer any questions.
