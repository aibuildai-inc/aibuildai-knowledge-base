# 14th Place Solution - Binary Classification on cropped frequency x time

Competition: rfcx-species-audio-detection
Rank: #14
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220725

First of all thanks to Kaggle and Hosts for organizing this competition.

We ( me & @ks2019 ) were initially working in Cassava Leaf Disease Classification but thanks to this [post](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/212610) by @cpmpml that gave us the direction that something different needs to be tried then what is going on in the public. On closer inspection, we found something similar to him. The frequency-time crops in the audios for a specie_id are almost constant (i.e. say for specie_id 23 - in most of the recordings audio frequency lies between 6459 and 11628, and its duration lasts for about 16 seconds). This gave us the idea of cropping out all the potential regions from the spectrogram and perform binary classification on them.

Our approach can be summarized as -
1. Crop images from spectrogram with frequency ranging between max-min frequency observed for a specie_id and with time duration 2 times the max duration observed for a specie_id 
2. Pre-Process: resize crops to size 128 x 256, scale between 0 and 1, and perform augmentation
3. Train B0 binary classifier detecting the presence of specie (a single binary classifier - here we tracked back using the frequency information of the crop that which ID we are asking classifier to detect for)
4. Generate Pseudo-labels
5. Retrain
4. Perform inference on the test, and take the mean of max n(in our case it was 3) probabilities observed for a specie_id in a recording as the probability of that specie_id

Note: On the very first submission, a single model with the above approach gave us 0.921 as public LB (has private LB 0.927), then pseudo labeling and a little bit of blending took private LB to 0.948

## Cropping
From each spectrogram, for each specie_id x songtype, we cropped out image sequences with the frequency range between min and max frequency observed for that specie_id x songtype, and then created image sequences with duration 2 times the max time interval the audio lasted for that specie_id x songtype in the train.
[Img]
[Here](https://github.com/PRATEEKKUMARAGNIHOTRI/CMS-trigger/blob/master/images/RFCX1.png) - In case img not visible

## Augmentation
Along with adding random noise, we took a false positive sample of the same specie_id and added that to the audio sample. After this augmentation, the label of the recording id x specie id remained the same(i.e. a false-negative remained the false negative and a true positive remained the true positive)
