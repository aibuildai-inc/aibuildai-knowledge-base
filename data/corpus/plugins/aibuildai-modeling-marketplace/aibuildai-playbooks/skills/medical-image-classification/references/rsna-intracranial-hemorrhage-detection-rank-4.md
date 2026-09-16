# 4th Place Solution with code

Competition: rsna-intracranial-hemorrhage-detection
Rank: #4
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/118249

Code: https://github.com/XUXUSSS/kaggle_rsna2019_4th_solution 
Our code is based on Appian's repo: https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage

# Overview of the proposed method


Our solution includes two stages. We train  2D CNN models in stage 1 for feature extraction, and 1D + 3D CNN models in stage 2 for classification.

## Preprocess
1.	Two window policies:
    a)	use Appian’s windowing policy
              i.	Three windows are: [40, 80], [80, 200], [40, 380][[link](https://github.com/XUXUSSS/kaggle_rsna2019_4th_solution/blob/1e9b6a5bb46d1d329f4af04e9066a3a0b7fa7769/IFE_1/src/cnn/dataset/custom_dataset.py#L68)]
    b)	Stack three consecutive slices to a 3-channel image. [[link](https://github.com/XUXUSSS/kaggle_rsna2019_4th_solution/blob/1e9b6a5bb46d1d329f4af04e9066a3a0b7fa7769/IFE_3/src/cnn/dataset/custom_dataset.py#L97)] 
              i.	Window: [40, 80] 
2.	Remove corrupted images 
3.	Filter out blank images by 
a)	Obtain the difference between maximum and minimum intensity value of each image, i.e., the intensity range, after applying a custom windowing scheme (center = 40, window = 80) 
b)	Remove images with intensity range &lt; 60 from both training and test sets. 
c)	The removed test images will be classified as negative during post-processing.
4.	Extract useful meta data from dicom files
a)	Patient ID
b)	StudyInstance ID
c)	SeriesInstance ID
d)	Position2
5.	Make patient-wise stratified five folds
a)	Images from one patient always belong to the same fold
b)	Class distributions are roughly the same across different folds

## STAGE 1: 2D Image Feature Extraction



### 1.	Training strategy:
a)	Randomly split the training dataset into 5 folds and train the model five times. Use 4 folds as training set and 1 fold as validation set each time.

### 2.	Models 
#### a)	EfficientNet B0
i.	ImageNet pretrained
ii.	Input image size: 512x512
iii.	Augmentation: random crop, random hflip, random rotate, random contrast
iv.	5-fold training
v.	TTA5： random crop, random hflip , random rotate, random contrast

#### b)	ResNext50 32x4d swsl 
i.	Semi-Supervised and Semi-Weakly Supervised ImageNet Models https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
ii.	Input image size: 448x448
iii.	Augmentation: random crop, random hflip , random rotate, random contrast, pixel and window jittering
iv.	5-fold training
v.     Cosine learning rate scheduler
vi.	TTA5： random crop, random hflip , random rotate, random contrast

### Summary of stage 1 models:


## STAGE1: Meta Data Feature Engineering


 
##STAGE2: Slice Sequence Model
In stage2,   we train 1D CNN model and 1D+3D CNN models for classification.
### 1D CNN model:


### 1.	Pipeline:
a)	Extract 1D feature and metadata from stage 1
b)	Stack the features that belong to one CT series together.
c)	Pass the stacked feature to customized fully convolutional neuronal networks, and generate the output. [[link](https://github.com/XUXUSSS/kaggle_rsna2019_4th_solution/blob/ed1c6f59b3077e3c8226671a5d9c38c2028aab5d/cls_2/src/cnn/models/model.py#L22)]
### 2.	Augmentation:
a)	No data augmentation
### 3.	Training strategy
a)	Follow 2D CNN’s fold split
### 1D+3D CNN model


### 1.	Pipeline:
a)	Extract metatdata,  1D and 3D features from stage 1.
b)	Stack the features that belong to one CT series together.
c)	Pass the stacked feature to customized fully convolutional neuronal networks, and generate the final output. [[link](https://github.com/XUXUSSS/kaggle_rsna2019_4th_solution/blob/ed1c6f59b3077e3c8226671a5d9c38c2028aab5d/cls_1/src/cnn/models/model.py#L141)]
### 2.	Augmentation:
a)	No data augmentation
### 3.	Training strategy
a)	Follow 2D CNN’s fold split
### Summary of Stage 2 models:



## Ensemble Predictions

4 x 5 x 5 = 100 Predictions from
1. 	4 Models
    a)	Cls_1a trained on Fold_Set_a,
    b)	Cls_1b trained on Fold_Set_b,
    c)	Cls_2 trained on Fold_Set_a,
    d)	Cls_3 trained on Fold_Set_c
2.	5 Folds per Fold Set
3.	5 TTA

## Post-processing
1. Assign the minimum value over all predictions to the blank test images
2. Clip the predicted value to the range of [1e-6, 1-1e-6]
3. Convert the predictions to the required submission format

## Score Growth Chart


Acknowledgement: Our code is based on Appian’s repo.  @appian Thank you very much for your great and beautiful work!
