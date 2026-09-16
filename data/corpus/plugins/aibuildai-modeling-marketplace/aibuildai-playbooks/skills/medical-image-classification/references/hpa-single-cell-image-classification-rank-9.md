# 9th Place Solution

Competition: hpa-single-cell-image-classification
Rank: #9
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238678

Hi,

We had a tough competition, congratulations to all the kagglers who persevered to the end and many thanks to the organizers and my wonderful teammates @daishu @garybios  @boliu0 

# Summary

We have two kinds of pipeline in this competition, they are:

* Pipeline1: Train on full image, test on single cells (mask out other cells)
* Pipeline2: Train on cropped cells, test on cropped cells also

Finally, we trained 20~ folds of model with these two pipelines and ensembled them by taking the mean value.

***Notice that we do not use image-level prediction.***

# Methods

### Pipeline1

We used two methods to train-test this pipeline.

The first one is to train with 512 images, and the test input is also 512. We loop n times for each image (n is the number of cells in the image), leaving only one cell in each time and masking out the other cells to get single cell predictions.

The second one is trained with 768 random crop 512, and then tested almost the same way as the first one, but not only mask out the other cells, but we also put the position of the cells left in the center of the image.

### Pipeline2

We pre-crop all the cells of each image and save them locally. Then during training, for each image we randomly select 16 cells. We then set bs=32, so for each batch we have 32x16=512 cells in total.

We resize each cell to 128x128, so the returned data shape from the dataloader is `(32, 16, 4, 128, 128)` . Next we reshape it into `(512, 4, 128, 128)` and then use a very common CNN to forward it, the output shape is `(512, 19)`

In the prediction phase, we will directly take this output and use it as the predicted value for each cell. ↑

But during the training process, we rereshape this `(512, 19)` prediction back into `(32, 16, 19)` . Then the loss is calculated for each cell with image-level GT label.


# Acknowledge

Special Thanks to Z by HP & NVIDIA for sponsoring me a Z8G4 Workstation with dual RTX6000 GPU and a ZBook with RTX5000 GPU!

Since I got the Z8G4 workstation and the ZBook last December, this is the third gold medal I've won from Kaggle ;)
