# 16th place solution

Competition: rsna-intracranial-hemorrhage-detection
Rank: #16
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117417

Congratulations to all winners and thanks to kaggle and organizers for opening this learning space.

We started relatively late, but we made a good starting point with the code that @appian shared. Great thanks to @appian 

Our overall procedure is as follows.



### In step 1
* Basic training is performed by considering an image as an independent input.
    * input shape : (batch_size, 512, 512, 3)
        * 4th axis (3) means 3 channels with multiple windowing parameters
    * output shape : (batch_size, 6)
    * CNN Architectures : SE-ResNeXt-101 and EfficientNet-B6
    * loss : weighted log loss (weights = [2/7, 1/7, 1/7, 1/7, 1/7, 1/7])
    * optimizer : Adam (with learning rate from 1e-4 to 1e-5)
    * sampling : random sampling or location based sampling (sampling middle slices more from image series in patient-level)
    * 5 folds or 7 folds training

### In step 2
* We wanted to calibrate the output distributions considering the relation of labels or adjacent image slices,  so we recognized the outputs of patient-level images as a signal and trained the model.
* Output distributions are extracted from the validation set. (For example, 5 models from 5 folds can make total training dataset.)
* If about 640,000 images are used in step1, about 19,500 output signals (the number of patients) are used in step 2.
    * input shape : (batch_size, None, 6, 1)
        * 1 axis (None) means the length of signal (the number of slices)
        * 2 axis (6) means the number of labels
    * output shape : (batch_size, None, 6, 1)
    * CNN Architecture : simple CNN model with 4 convolution layers having 5x6 matrix
    * loss : weighted log loss (weights = [2/7, 1/7, 1/7, 1/7, 1/7, 1/7])
    * optimizer : Adam (with learning rate 1e-5)
    * 5 folds training

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, None, 6, 1)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, None, 6, 64)       1984      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, None, 6, 64)       122944    
_________________________________________________________________
conv2d_3 (Conv2D)            (None, None, 6, 64)       122944    
_________________________________________________________________
conv2d_4 (Conv2D)            (None, None, 6, 64)       122944    
_________________________________________________________________
conv2d_5 (Conv2D)            (None, None, 6, 1)        65        
=================================================================
```

We also thought to handle sequential information in image-level, but the deadline was short, so the process was split into two steps and output signals with relatively small dimensions were used as the next best thing. 

**The results are as follows.**
**step 1 result : 0.05425 (private score)**
**step 2 result : 0.04793 (private score)**
**We think that the core processing of our team, like other teams, also was to reflect sequential information.**
