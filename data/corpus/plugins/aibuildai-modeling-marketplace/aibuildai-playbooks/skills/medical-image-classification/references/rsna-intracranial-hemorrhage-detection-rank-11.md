# 11th place solution (with updated code on github)

Competition: rsna-intracranial-hemorrhage-detection
Rank: #11
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117330

Congratulations to all. 
Thank you kaggle and the host team for organizing this interesting competition.

The updated source code is available at https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage
I will probably upload all trained models later.

### Windowing

For this challenge, windowing is important to focus on the matter, in this case the brain and the blood. There are good kernels explaining how windowing works.

- [See like a Radiologist with Systematic Windowing](https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing) by [David Tang](https://www.kaggle.com/dcstang)
- [RSNA IH Detection - EDA](https://www.kaggle.com/allunia/rsna-ih-detection-eda) by [Allunia](https://www.kaggle.com/allunia)

We used three types of windows to focus and assigned them to each of the chennel to construct images on the fly for training.

| Channel | Matter | Window Center | Window Width |
----------|--------|---------------|---------------
| 0 | Brain | 40 | 80 |
| 1 | Blood/Subdural | 80 | 200 |
| 2 | Soft tissues | 40 | 380 |

Here is an example before and after applying the windowing. This image is labeled as `any intraparenchymal` and you can see that windowing helps focusing on the matter. Please check [windowing.ipynb](https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage/blob/master/demo/dicom_windowing.ipynb) for the detail.

[windowing.png]


### Classification

This step focuses on pixel data contained in DICOM file not meta data. But still four kind of meta data is used to apply windowing properly. `RescaleSlope` and `RescaleIntercept` are used for windowing. `BitsStored` and `PixelRepresentation` are used for fixing wrong intercept values which is mentioned in [Cleaning the data for rapid prototyping](https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai) written by [Jeremy Howard](https://www.kaggle.com/jhoward). 

- Two architectures are used. `se_resnext50_32x4d` and `se_resnext101_32x4d`. 
- Imagenet pretrained weights from https://github.com/Cadene/pretrained-models.pytorch
- 8 folds each. 
- Adding a random number to windowed pixel data as augmentation led to a little better generalization performace. This idea is based on a hunch that CT scanners are probably not perfectly calibrated. 
- Test time augmentations(n=5) are used for predictions.
- Checkpoints from 2nd and 3rd epochs are used for predictions and then averaged.
- Final predictions are obtained from simple average of `se_resnext50_32x4d` and `se_resnext101_32x4d`. 

**The training result of 0th fold of se\_resnext50\_32x4d ([model100.py](https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage/blob/master/conf/model100.py))**

[model100_fold0.png]

**The training result of 0th fold of se\_resnext101\_32x4d ([model110.py](https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage/blob/master/conf/model110.py))**

[model110_fold0.png]

**Logloss for each of the Hemorrhage Types after emsembling (oof)**

[ensembled.png]

This ensembled score (0.0642) is similar to the score (0.065) we got on public LB in the first stage before introducing second level model.


### Second Level Model

The second level model focuses on a series of CT scan unlike the classification model which focuses on a given image(slice). The main idea is that other slices of a certain slice within the same series can be useful to enhance the predictions of that slice. For example, if both of the adjacent slices of a certain slice are inferred as `epidural`, the middle of the slice is most likey `epidural`. This kind of relationships can trained using something like LightGBM. The train data can be constructed as follows,

For example, in case of training `epidural` based on oof predictions, you can construct a record like this,

```
prediction of the given slice, left1, right1, left2, right2, left3, right3, ...,
```

- `left1` indicates the prediction of the first slice to the left from the given slice.
- `right2` indicates the prediction of the second slice to the right from the given slice.

We included `left1` to `left9` and `right1` to `right9` for each of the slice. `left1` and `right1` are unsurprisingly the most useful features among all other slices except the given slice based on feature importance(lightgbm gain). Some distant slices such as `left9` or `right9` are not as important as closer slices to the given slice but still somewhat useful.

[secondlevel.png]

- Final predictions are obtained by simply averaging predictions from LightGBM, Catboost and XGB.
- The 1st stage public LB score was improved from 0.65 to 0.57 by this.

Thank you for reading!
