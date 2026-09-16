# 10th Place Solution

Competition: rsna-2023-abdominal-trauma-detection
Rank: #10
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447450

Congrats to all prize and medal winners! I enjoyed this competition because there are a great variety of options for solving this task as with last year's RSNA competition. We share our team's solution ( @ren4yu + @tattaka ).

# Summary
Our solution is to first segment the organs, cut out each organ region, and build a dedicated model for each organ.
For the bowel and extravasation classes, for which large regions must be explored, we do not perform segmentation to cut out the regions, but instead perform simple black region removal and input the large regions into the models.
The results of each model are refined by the stacking model and submitted as the final result.



# Segmentation Model
We used the 3D SwinUNETR model provided by MONAI. It works surprisingly well with even a small amount of training data. To reduce the computational cost, the entire voxel was resized into 128x128x128 before segmentation.


# Organ Models
For liver, spleen, kidney, and bowel cropped regions, 2.5D CNN + LSTM models are used.

## Liver and Spleen Models
Cropped region is resized into 16x386x386, and fed into dedicated models.

## Kidney Model
Left and right kidney are independently cropped, resized, and concatinated along with horizontal axis. This enables us horizontal flip augmentation and TTA. Concatenated region becomes 16x224x448.

## Bowel Model
Cropped region is resized into 64x224x224. In training bowel model, both patient level label and image level labels were used.

# Stacking Model
The purpose of the stacking model is to directly optimize the average of the weighted logloss, which is the metric for this competition, including the logloss for any_injury. Each model is optimized for the weighted logloss for each of the injury type, but not for any_injury because it is automatically calculated from the probabilities of the other injuries. For any_injury, it is essential to optimize any_injury because the weight for any_injury is relatively large.
As a stacking model, we use a simple 4-layer MLP, and trained with competition metric as loss function.

The table below shows the CV evaluation results before and after stacking (The values below are out of date because we wrote this solution before the competition was extended. The final submission's CV is 0.3686).

|                | bowel    | extravasation | kidney  | liver   | spleen  | any     | overall |
|----------------|----------|---------------|---------|---------|---------|---------|---------|
| w/o stacking   | 0.1186   | 0.4716        | 0.2948  | 0.4282  | 0.4301  | 0.5528  | 0.3827  |
| with stacking  | 0.1034   | 0.4885        | 0.2861  | 0.4409  | 0.4452  | 0.4777  | 0.3736  |


# tattaka's Part
I was in charge of bowel and extravasation classification. Our solution followed [the 2-stage approach of the RSNA competition 3 years ago](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/discussion/194145).  
The basic setup is as follows

## Image-Level Modeling (1st stage)
The input for the 1st stage is a 3-channel image including adjacent frames.  
1epoch training was performed on all labeled images.  
* backbone: resnetrs50  
* head: 
  * Separate the head by bowel and extravasation
  ```python
  nn.Sequential(
      nn.Conv2d(num_features[-1], 512, kernel_size=1),
      nn.AdaptiveAvgPool2d((1, 1)),
      Flatten(),
      nn.Dropout(0.5, inplace=False),
      nn.Linear(512, 1),
  )
  ```
In the 1st stage, students learn bowel and extravasation at the same time.  

## Series-Level Modeling (2nd stage)
The input for the 2nd stage also followed the previous solution.  
Use the 512 dimensions after Flatten in the head of the 1st stage as image features.
Image features are created with stride=3 instead of all images, and the input sequence length is set to a maximum of 256 in the same way as the previous solution.   
The differences between adjacent features are combined, and the input to the model is in the form (bs, 256, 1536). 
* model:
  * Combines the attention pooling and max pooling of the BiGRU outputs in one layer to create an series-level prediction.
  * The BiGRU output should also predict the image label. 
  
Unlike the 1st stage, bowel and extravasation were optimized separately.

## Tricks for Successful Training
There are a few tricks to successful learning in this competition.

* Because of large data imbalance, upsampling of the positive sample by a factor of 10 (high impact)
  * In addition, focal loss is used
* After training stage1 at image level, max of the logit and gt are trained again as labels for the new image level (high impact)
  * repeated it twice
  * Perhaps the image label is noisy
* Rule-based removal of the outer black areas before the image is entered into the model
  * Because of the longer computation time when using a larger resolution, a size of 384x384 is used after removing the outer area.

```python
def region_crop(img: np.ndarray) -> np.ndarray:
    image_1ch = (img.mean(2) * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    image_1ch = cv2.erode(image_1ch, kernel, iterations=2)
    mask = image_1ch > 50
    if mask.sum() == 0:
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    if (y_max - y_min) > 5 and (x_max - x_min) > 5:
        img = img[y_min:y_max, x_min:x_max]
    return img
```
The bowel and extravasation scores before stacking are
|        | bowel logloss | bowel auc | ev logloss | ev auc |
| ------ | ------------- | --------- | ---------- | ------ |
| stage1 | 0.2719        | 0.9314    | 0.4602     | 0.8287 |
| stage2 | 0.1167        | 0.9167    | 0.4579     | 0.8264 |

## Not works
* label smoothing
* resnet3dcsn
  * Not bad, but I didn't have time to tune in.
* scaling logit
* GeM pooling
* Other backbone
  * ConvNeXt is slightly worse than resnetrs50
  * I could not get the backbone of the transformer to work
