# 5th Place Solution Overview

Competition: rsna-str-pulmonary-embolism-detection
Rank: #5
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193475


**Code base** : https://github.com/darraghdog/rsnastr 
**Kaggle submission** : https://www.kaggle.com/darraghdog/rsnastr2020-prediction

Thanks to all the organisers for hosting this competition. And congratulations to all the participants and winners - these competitions and solutions are getting more and more advanced 😁  

My compute was a v100 on AWS initially with 320X320 images, with prototyping on Macbook (a lot can be checked on CPU). Then in the last 3 weeks I moved to an A100 to run 512X512 (thanks DoubleYard) - 40GB GPU memory per card - one card was enough. 
Overall the solution pipeline ended up being pretty similar to last year, but the journey to get there was a lot different. The metric used by the competition was difficult to simulate, and all parts of the pipeline needed to be finalised/integrated to get feedback on the competition benchmark.  

**Preprocessing**
Use the [windowing](https://github.com/darraghdog/rsnastr/blob/14c8516d3a81bc26c5101afff004ce1e99c3f5a9/preprocessing/dicom_to_jpeg.py#L59-L75) from Ian Pan's [post](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930) pretty much as is, just [parallelised](https://github.com/darraghdog/rsnastr/blob/14c8516d3a81bc26c5101afff004ce1e99c3f5a9/preprocessing/dicom_to_jpeg.py#L77-L118) it to speed it up. Some dicom's fell out which I saw later was due to the way pydicom was used, but I think the majority were good. 
Light [augmentations](https://github.com/darraghdog/rsnastr/blob/14c8516d3a81bc26c5101afff004ce1e99c3f5a9/training/pipeline/train_image_classifier.py#L72-L74), maybe more would have helped but did not want to lose information which side the PE lay on. 

**Image Level**
Efficientb5 seemed to be better than anything else I tried both in terms of speed and loss. Given the time and compute limits on submission, speed was important. 
The datasampler took at least [two images from each study, in each epoch, and positive images were oversampled](https://github.com/darraghdog/rsnastr/blob/14c8516d3a81bc26c5101afff004ce1e99c3f5a9/training/pipeline/train_image_classifier.py#L112-L117) to give a rate of 4:1 -ve:+ve, with image batchsize of 48 (using half point precision - amp). 
[Weighted study loss](https://github.com/darraghdog/rsnastr/blob/14c8516d3a81bc26c5101afff004ce1e99c3f5a9/training/pipeline/train_image_classifier.py#L132-L138) was calculated at each step on the positive samples only (weights as per competition metric weights excl. `negative_exam_for_pe`), image loss was calculated on all samples. Final loss per step was [sum of both averages](https://github.com/darraghdog/rsnastr/blob/14c8516d3a81bc26c5101afff004ce1e99c3f5a9/training/pipeline/train_image_classifier.py#L200-L209); image loss was tuned so that image loss and study loss would be roughly equal. 
Each epoch took about 5 mins and ran ~15 epochs, final solution used three of five folds.

**Study Level**
Extracted gap layer of each image to disk. These were fed into [three independent sequence models](https://github.com/darraghdog/rsnastr/blob/14c8516d3a81bc26c5101afff004ce1e99c3f5a9/training/pipeline/train_sequence_classifier.py#L113-L124). 
1) 2X Bi-LSTM, more or less [same architecture](https://github.com/darraghdog/rsna) as last year.
2) Bert transformer model with 1 layer. 
2) Bert transformer model with 2 layers. 
Each of these used the same loss [(or at least as close as I could get)](https://github.com/darraghdog/rsnastr/blob/14c8516d3a81bc26c5101afff004ce1e99c3f5a9/training/pipeline/train_sequence_classifier.py#L150-L170) as the competition metric on each step, study batchsize 64, no accumulation (again with amp).
Finally both models and all folds were averaged. 
I submitted a few more transformer models on last day which I had high hopes for but most broke the consistency check - I only allowed submission, if it passed - and then time ran out. (Thanks @yuval6967 for the tip to only drop to disk if it passed). Actually I think 7 of my last 10 submissions failed. 

**Inference**
A few tricks to help speed up and bring RAM and GPU memory down - [final sub](https://www.kaggle.com/darraghdog/rsna512-effnetb5-fold-all-exam-xfrmr-validated?scriptVersionId=45523025).
Batchsize of 1. Feed sequences of images into image level model in chunks of 128 images, but only normalize (uint8->float32) when that chunk is to be passed into image model. Then concat all output (GAP layer) chunks to make the sequence.  
Convert all models and input data to half point precision `model=model.half()` - this halved the memory, but results were just as good. 
With the above, I managed to squeeze three folds out of the above pipeline - four folds also worked, but it failed on consistency check on last day. Interested to see optimisations from other competitiors. 

A final note, about two weeks ago, I was stuck on CV could not get the metric working, and decided to kick the competition. Luckily I stuck with it to try a first submission which got in at LB 0.181 single fold :) For such a challenging metric and large dataset the timeline was tight, I think with another couple of weeks the community could have got to 0.12x or so on the leaderboard - but maybe that would overfit the dataset.
