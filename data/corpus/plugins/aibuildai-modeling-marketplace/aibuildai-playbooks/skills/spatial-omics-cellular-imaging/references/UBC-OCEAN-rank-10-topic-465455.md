# 10th place solution

Competition: UBC-OCEAN
Rank: #10
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465455

This was an interesting competition and I would like to thank my teammate @samfc10 and everyone involved with the organization of it.

This is a simple textbook solution that heavily relies on external TMA data and strong labels. There is nothing special or novel in this pipeline.

* [Inference](https://www.kaggle.com/code/gunesevitan/ubc-ocean-inference)
* [libvips/pyvips Installation and Getting Started](https://www.kaggle.com/code/gunesevitan/libvips-pyvips-installation-and-getting-started)
* [UBC-OCEAN - JPEG Dataset Pipeline](https://www.kaggle.com/code/gunesevitan/ubc-ocean-jpeg-dataset-pipeline)
* [UBC-OCEAN - EDA](https://www.kaggle.com/code/gunesevitan/ubc-ocean-eda)
* [UBC-OCEAN - Dataset](https://www.kaggle.com/datasets/gunesevitan/ubc-ocean-dataset)
* [GitHub Repository](https://github.com/gunesevitan/ubc-ovarian-cancer-subtype-classification-and-outlier-detection)

## 1. Raw Dataset

### WSI

Masks of WSIs are resized to thumbnail sizes. Tiles of WSIs and masks are extracted from their thumbnails with stride of 384 and they are padded to 512. A MaxViT Tiny FPN model is trained on those padded tiles and masks. Segmentation model outputs are activated with sigmoid and 3x TTA (horizontal, vertical and diagonal flip) are applied after the activation.

Final segmentation mask prediction is blocky since the model was trained on tiles and merged later.

[seg1]

Segmentation mask predictions are cast to 8-bit integer and upsampled to original WSI size with nearest neighbor interpolation.

[seg2]

* WSI and their mask predictions are cropped maximum number of times with stride of 1024.
* Crops are sorted based on their mask areas in descending order
* Top 16 crops are taken and WSI label is assigned to them

### TMA

Rows and columns with low standard deviation are dropped on TMAs with the function below. The purpose of this preprocessing is removing white regions and making WSIs and TMAs as similar as possible. Using higher values of threshold were dropping areas in the tissue region so the standard deviation threshold is set to 10.

```
def drop_low_std(image, threshold):

    """
    Drop rows and columns that are below the given standard deviation threshold

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, 3)
        Image array

    threshold: int
        Standard deviation threshold

    Returns
    -------
    image: numpy.ndarray of shape (cropped_height, cropped_width, 3)
        Cropped image array
    """

    vertical_stds = image.std(axis=(1, 2))
    horizontal_stds = image.std(axis=(0, 2))
    cropped_image = image[vertical_stds > threshold, :, :]
    cropped_image = cropped_image[:, horizontal_stds > threshold, :]

    return cropped_image
```

[seg2]

## 2. Validation

Multi-label stratified kfold is used as the cross-validation scheme. Dataset is split into 5 folds. `label` and `is_tma` columns are used for stratification.

## 3. Models

EfficientNetV2 small model is used as the backbone with a regular classification head.

## 4. Training

CrossEntropyLoss with class weights are used as the loss function. Class weights are calculated as n / n ith class.

AdamW optimizer is used with 0.0001 learning rate. Cosine annealing scheduler is used with 0.00001 minimum learning rate.

AMP is also used for faster training and regularization.

Each fold is trained for 15 epochs and epochs with the highest balanced accuracy are selected.

Training transforms are:

* Resize TMAs to size 1024 (WSI crops are already 1024 sized)
* Magnification normalization (resize WSI to 512 and resize it back to 1024 with a random chance)
* Horizontal flip
* Vertical flip
* Random 90-degree rotation
* Shift scale rotate with 45-degree rotations and mild shift/scale augmentation
* Color jitter with strong hue and saturation
* Channel shuffle
* Gaussian blur
* Coarse dropout (cutout)
* ImageNet normalization

## 5. Inference

5 folds of EfficientNetV2 small model are used in the inference pipeline. Average of 5 folds are taken after predicting with each model.

3x TTA (horizontal, vertical and diagonal flip) are applied and average of predictions are taken.

16 crops are extracted for each WSI and average of their predictions are taken.

The average pooling order for a single image is:
* Predict original and flipped images, activate predictions with softmax and average
* Predict with all folds and average
* Predict all crops and average if WSI 

## 6. Change of Direction

The model had 86.70 OOF score (TMA: 84, WSI: 86.59) at that point but the LB score was 0.47 (private 0.52/32th-42th) which was very low.

[wsi_confusion_matrix1]

[tma_confusion_matrix1]

[confusion_matrix1]

I noticed some people were getting better LB scores with worse OOF scores and I was stuck at 0.47 for a while. I had worked on Optiver competition for 2 weeks and came back. I decided to dedicate my time to finding external data because breaking the entire pipeline and starting from scratch didn't make sense.

## 7. External Data

### UBC Ocean
The most obvious one is the test set image that is classified as HGSC confidently. 16 crops are extracted from that image and HGSC label is assigned to them.

### Stanford Tissue Microarray Database

134 ovarian cancer TMAs are downloaded from [here](https://tma.im/cgi-bin/viewArrayBlockList.pl).

Classes are converted with this mapping

```
CLASS_MAPPING = {
    'fibroma of ovary spindle cell fibroma of ovary': 'Other',
    'carcinoma papillary serous': 'HGSC',
    'carcinoma endometrioid': 'EC',
    'lymphoma precursor B lymphoblastic': 'Other',
    'carcinoma adeno': 'HGSC',
    'carcinoma clear cell': 'CC',
    'carcinoma mucinous': 'MC',
    'carcinoma adeno mucinous': 'MC',
    'seminoma dysgerminoma': 'Other'
}
```

### kztymsrjx9

This dataset is downloaded from [here](https://data.mendeley.com/datasets/kztymsrjx9/1). HGSC label is assigned to images in the Serous directory. Images in the Non_Cancerous directory are not used. 398 ovarian cancer TMAs are found here.

### tissuearray.com

Screenshots of high resolution previews are taken from [here](https://www.tissuearray.com/tissue-arrays/Ovary). 1221 ovarian cancer TMAs are found here.

### usbiolab.com

Screenshots of high resolution previews are taken from [here](https://usbiolab.com/tissue-array/product/ovary). 440 ovarian cancer TMAs are found here.

### proteinatlas.org

Images are downloaded from [here](https://www.proteinatlas.org/search/prognostic:ovarian+cancer;Favorable+AND+sort_by:prognostic+ovarian+cancer). 376 ovarian cancer TMAs are found here.

### Summary

Those were the sources where I found the external data.

|                                     | Images | Type | HGSC | EC  | CC  | LGSC | MC  | Other |
|-------------------------------------|--------|------|------|-----|-----|------|-----|-------|
| UBC Ocean Public Test               | 16     | WSI  | 16   | 0   | 0   | 0    | 0   | 0     |
| Stanford Tissue Microarray Database | 134    | TMA  | 37   | 11  | 4   | 0    | 4   | 78    |
| kztymsrjx9                          | 398    | TMA  | 100  | 98  | 100 | 0    | 100 | 0     |
| tissuearray.com                     | 1221   | TMA  | 348  | 39  | 24  | 140  | 100 | 570   |
| usbiolab.com                        | 440    | TMA  | 124  | 40  | 29  | 89   | 68  | 90    |
| proteinatlas.org                    | 376    | TMA  | 25   | 155 | 0   | 63   | 133 | 0     |

## 8. Final Iteration

Final dataset (including 16 crops per WSI) label distribution was like this

* HGSC: 4127
* EC: 2252
* CC: 1666
* MC: 1066
* LGSC: 969
* Other: 738

and image type distribution was like this

* WSI (16x 1024 crops): 8224
* TMA: 2594

All the external data are concatenated to each fold's training sets. Validation sets are not changed in order to get comparable results. OOF score is decreased from 86.70 to 83.85 but LB score jumped to 0.54. I thought this jump was related to Other class but the improvement wasn't good enough. That's when I thought private test set could have more Other classes which is very likely of Kaggle competitions. Twist of this competition was predicting TMAs and Other so private test set would likely have more of them. I decided to trust LB and selected a submission with the highest LB score. That submission scored 0.54 on public and 0.58 on private.
