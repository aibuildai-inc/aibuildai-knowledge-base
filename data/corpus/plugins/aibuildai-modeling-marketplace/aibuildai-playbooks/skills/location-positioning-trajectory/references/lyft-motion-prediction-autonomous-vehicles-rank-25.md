# 25nd place Summary

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #25
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199540

## Tools I used
- Tensorflow/Keras
- Colab (with TPU, free version)
- Google Cloud Storage
     I rendered the data offline, packed the data into .tfrec format, and uploaded them to Google Cloud Storage (~2TB), To ensure that the size would not be exploded, I compress the image into PNG format

## Dataset
- Used training (not full version*) and validation data
- Used default setting (from sample code) for rendering except min_future_frame ( = 10)

## Network backbone
- Xception
     I had tried a lot of backbone provided from Keras and different version of Resnet (101, 50, 34, 18), and found Xception's quality was the best (through validation)
- Batch size = 256

## Other tricks
- Validation
    Validation data and part of training data were used, and I separated both training and validation data into five parts by y-position of agent's center points. I made sure that the same (or similar) maps would not appear in both training and validation data simultaneously
- Multimode
     I predicted 4 modes and picked the top-3 (determined by confidences) modes. Besides, I had tried more modes (> 4) but found only 4 modes had larger confidence values while the other modes' never be picked. The unbalanced issue seems common in multimode trick
- Label smoothing  
     I "multiply" a small gaussion noise (mean = 1, std = 0.00333) to each ground truth.
- Custom loss function
     I used the evaluation metric for multimode (provided from official) as loss function. To gave more penalty to the corner cases, the loss was multiply by a weight, which was proportional to  the "Angle" of each agent
     Angle = acos(x/sqrt(x**2 + y ** 2)), (x, y) is the coordinate of the final position of each agent 
- Discard History
     I used 10 history frames for training, but I found that around 5% testing data had history frames less than 10. So, I discard the history of target history frame randomly

*Due to the limitation of memory in colab, I could not use preprocess full training data. So sad.
