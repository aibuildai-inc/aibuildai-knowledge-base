# 2nd Place Solution - Sequential model

Competition: rsna-intracranial-hemorrhage-detection
Rank: #2
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117228

Code &amp; Val/LB scores : https://github.com/darraghdog/rsna    
Congrats all winners, looking forward to go through your solutions. Big shout out to competition hosts RSNA, kaggle community, pytorch community, albumentations and FB's work on resnext -very cool how they trained this. 
We were very sad not to get a top3 in Recursion competition, now we are very happy  😄  
    

     
**Overview**
In general we just have a single image classifier, data split on 5 folds, we only trained on 3 of them, and then extracted pre-logit layer from the classifier and fed into an LSTM.
Classifier trained on 5 epochs each fold, 480 images with below pre-processing. Each epoch, each fold, we extract embedding layer (use TTA and avg embeddings) train a separate LSTM for 12 epochs on each of those - so 15 LSTMs (3 fold image models X 5 epochs), and average the predictions. 
Was a bit concerned the preprocessing filter may lose information, so trained the above again without the preprocessing filter and it did worse; but averaging both pipelines did ever so slightly better. The pipeline from first paragraph above would, for all intensive purposes be just as good as final solution, but as we needed to fix docu pre-stage 2 the two pipelines are in github and final solution.  

**Preprocessing:**
- Used Appian’s windowing from dicom images. [Linky](https://github.com/darraghdog/rsna/blob/master/eda/window_v1_test.py#L66)
- Cut any black space. There were then headrest or machine artifacts in the image making the head much smaller than it could be - see visual above. These were generally thin lines, so used scipy.ndimage minimum_filter to try to wipe those thin lines. [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainorig.py#L159)
- Albumentations as mentioned in visual above. 

**Image classifier**
- Resnext101 - did not spend a whole lot of time here as it ran so long. But tested SeResenext and Efficitentnetv0 and they did not work as well. 
- Extract pre logit layer (GAP layer) at inference time  [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainorig.py#L387) 

**Create Sequences**
- Extract metadata from dicoms :  [Linky](https://github.com/darraghdog/rsna/blob/master/eda/meta_eda_v1.py) 
- Sequence images on Patient, Study and Series - most sequences were between 24 and 60 images in length.  [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainlstmdeltasum.py#L200) 

**LSTM**
- Feed in the embeddings in sequence on above key - Patient, Study and Series - also concat on the deltas between current and previous/next embeddings (`current-previous embedding` and `current-next embedding`) to give the model knowledge of changes around the image.  [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainlstmdeltasum.py#L133) 
- LSTM architecture lifted from the winners of first stage toxic competition. This is a beast - only improvements came from making the hiddens layers larger. Oh, we added on the embeddings to the lstm output and this helped a bit also.  [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainlstmdeltasum.py#L352) 
- For sequences of different length, padded them to same length, made a dummy embedding of zeros, and then through the results of this away before calculating loss and saving the predictions.  

**What did not help...**  
Too long to do justice... mixup on image, mixup on embedding, augmentations on sequences (partial sequences, reversed sequences), 1d convolutions for sequences (although SeuTao got it working)

**Given more time**  
Make the classifier and the lstm model single end-to-end model. 
Train all on stage2 data, we only got to train two folds of the image model on stage-2 data.
