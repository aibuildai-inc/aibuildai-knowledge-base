# 13th place solution with code

Competition: rsna-intracranial-hemorrhage-detection
Rank: #13
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/119079

Thanks Kaggle and RSNA for hosting such an interesting competition.
Thanks the whole team @andy2709 @moewie94 @lego1st @nguyenbadung for the great collaboration.

The code is publicly available at https://github.com/dattran2346/rsna-2019

## 1. Preprocessing

### Windowing

- We use various combination of  brain, subdural, bony, and default window and stack them to create a normal 3-channel image.

```
ct_windows = {
    'brain': {'L': 40, 'W': 80},
    'subdural': {'L': 75, 'W': 215},
    'bony': {'L': 600, 'W': 2800},
    'default': { # from metadata }
}
```

### Window setting optimization
- The idea is to use a 1x1 convolution and sigmoid activation to learn relevant windows, the weight is initialized to be the default brain, subdural and bony window. @andy2709 tried this method and noticed that the final learned window is very closed to the default window. 

### Data split
- We splited the dataset by both patient id and study id. I, @andy2709 and @lego1st trained the models by patient split, while @nguyenbadung and @moewie94 trained by study split.

## 2. Model
### 2D Model


We applied 2 stage training here:
- In the 1st stage, just a normal CNN training,  backbones are EfficientNetB2-B5, SEResNeXt50, SEResNeXt101.
- In the 2nd, we use 5 consecutive slices’ outputs and applied a simple CNN to predict the center slice.

### 3D Model


- We use a normal backbone as a decoder and a bi-directional LSTM with a FC layer as the decoder, the model was trained end-to-end.
- For each study, we select 10 random slices (contiguous and not) in order and put though the network during training. For inference, all slices are considered.

## 3. Stacking


- Concatenate prediction from all model (split by both study-id and patient-id), build simple cnn model:

```
model = nn.Sequential(
          nn.Linear(input_dim, 1024),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(1024, 1024),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(1024, 6),)
```

- Average prediction from 2 types of model:  study id split and patient id split.

## 4. Summary

