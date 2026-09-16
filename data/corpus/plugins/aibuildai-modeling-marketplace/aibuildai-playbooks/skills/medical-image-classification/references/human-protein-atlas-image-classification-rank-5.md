# 5th Place Solution

Competition: human-protein-atlas-image-classification
Rank: #5
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77731

First of all, congratulations to all the winners! Thanks to Kaggle and HPA team for hosting such an interesting competition and thanks to  [TomomiMoriyama](https://www.kaggle.com/tomomimoriyama), [Heng CherKeng](https://www.kaggle.com/hengck23), [ManyFoldCV](https://www.kaggle.com/manyfoldcv) and [Spytensor](https://www.kaggle.com/spytensor).  

Here is a brief summary of our solution.

### DataSet

Like most other competitors, we used both official (both PNG and TIFF) and external data. To deal with class-imbalance, we used WeightedRandomSampler （method in pytorch） during training and [MultilabelStratifiedShuffleSplit](https://github.com/trent-b/iterative-stratification) to split the data into training and validation. We constructed 10 folds cross validation sets with 8% for validation.

### Image Preprocessing

The HPA dataset has four dyeing modes each of which is an RGB image of its own, so we took only one channel （r=r,g=g,b=b,y=b） to form a 4-channel input for training.

All PNG images are kept at their original 512 size, whereas the TIFF images are resized to 1024.

### Augmentation

Rotation, Flip, and Shear.

We didn't use random cropping. Instead we trained 5 models using crop5 (method in pytorch) and found it to be more effective.

### Models

For our base networks, we mainly used Inception-v3，-v4, and Xception. We have also tried DenseNet, SENet and ResNet, but the results were suboptimal.

We used three different scales during training (512 for PNG images and 650, 800 for TIFF images) with different random seeds for the 10-folds CV.

Modifications

1. Changed the last pooling layer to global pooling.
2. Appended an additional fully connected layer with output dimension 128 after the global pooling.
3. We also divided the training process into two stages where the first stage used size 512 with model trained on ImageNet, and the second stage used size 650 or 800 with model trained from the first stage. We found this to be slightly better than training with fixed size all the way.

### Training

* loss: [MultiLabelSoftMarginLoss](https://pytorch.org/docs/stable/nn.html?highlight=multilabelsoftmarginloss#torch.nn.MultiLabelSoftMarginLoss)
* lr: 0.05 (for size 512, pretrained on ImageNet)，0.01 (for size 650 and 800，pretrained using size 512); lrscheduler: steplr(gamma=0.1,step=6)
* optimizer: SGD
* epochs: 25, early stopping for training with size 650 or 800 (around 15 epochs), model selected based on loss (instead of F1 score)
* sampling weights for different classes: [1.0, 5.97, 2.89, 5.75, 4.64, 4.27, 5.46, 3.2, 14.48, 14.84, 15.14, 6.92, 6.86, 8.12, 6.32, 19.24, 8.48, 11.93, 7.32, 5.48, 11.99, 2.39, 6.3, 3.0, 12.06, 1.0, 10.39, 16.5]
  
### Multi-Thresholds

We used the validation sets to search for threshold for each class by optimizing the F1 score begining with 0.15 for all classes.


### Test

(with multi-thresholds)



### Ensembling

Final prediction is ensemble of above methods: Size 800, 10-fold for Inception-v3; Size 650 and 800, 10-fold for Inception-v4; Size 800, 10-fold, Size 650, 1-fold, Size 512, 5-fold for Xception (the reason for 5-fold instead of 10 was simply because we didn't have enough submissions to check the performances of all models, so we simply took the best ones).

### Things that did not work for us
* Training with larger input size (&gt;= 1024), which forced us reduce the batch size.
* 3-channel input
* focal loss
* C3D
* TTA: unlike a lot of other competitors, TTA during test time actually didn't work for us.
* Other traditional machine learning methods such as DecisionTree, RandomForest, and SVM.
