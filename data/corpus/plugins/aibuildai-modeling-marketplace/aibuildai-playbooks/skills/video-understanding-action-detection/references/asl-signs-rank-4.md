# 4th place solution

Competition: asl-signs
Rank: #4
Source: https://www.kaggle.com/c/asl-signs/discussion/406673

I would like to thank the organizers and all the competitors.   
The overall picture of my solution is shown in the figure below.



# Preprocessing
+ Use XY coordinates
+ Normalize the coordinates between the eyebrows to (0,0).
+ Compare the number of frames detected for the right and left hands, and flip only when the number of frames detected for the right hand is less than that of the left hand (x=-x).
+ Use XY coordinates of 21 feature points of the right hand (flip left hand) and 40 feature points of the lips, for a total of 122 dimensions.
+ Delete frames in which the feature points of the hand have not been detected. Therefore, the number of frames for the hand and the number of frames for the lips will be different."

# Modeling
+ I have constructed two types of 1D CNNs. 
+ The first is a model that classifies fixed-length sequences (1DCNN-FixLen). 
+ The second is a model that classifies variable-length sequences (1DCNN-VariableLen). 
+ Both models have a hand backbone, a lips backbone, and a head for classification. 
+ The hand and lips backbones are similar, except that the convolution dimension differs: 128 for the hand backbone and 64 for the lips backbone.
## 1DCNN-FixLen
+ Interpolate the sequence to 96. This process is done for both hand and lip feature points.
### Backbone
+ Apply conv 11 times and max_pooling 3 times to reduce sequence length to 12 (=96/(2**3)).
+ Apply conv to increase the dimension to 512, then apply global_max_pooling.
  + Changing global_max_pooling to global_avg_pooling resulted in a significant performance drop. This is probably because important features in the sequence are only included in a few frames. We believe that with global_avg_pooling, these features are averaged out and disappear.
  + I tried other gating mechanisms such as Gated Linear Unit, but global_max_pooling was the best.
### Head
+ Sum the hand features (512 dim) and lips features (512 dim).
+ Apply 6 fully connected layers (shallow version, stochastic_depth=0.1) or 18 fully connected layers (deep version, stochastic_depth=0.5).
  + The deep version ties the parameters of the fully connected layers to prevent model bloat.
+ Classify into 250 classes.
## 1DCNN-VariableLen
### Backbone
+ Apply conv (kernel_size=3) 5 times in total and conv (kernel_size=1) 6 times in total. Unlike above, pooling is not applied.
    + Therefore, the size of the receptive field is 11 frames. Since reducing the number of frames further worsened the situation, and increasing the number of frames worsened the situation, it seems that the meaningful sequence was about 11 frames, even if the sequence was long.
+ Apply global_max_pooling after increasing the dimension to 512 with conv.
    + During training, the output is masked by sequence length and global_max_pooling is applied. This is because 0-padding is applied to normalize by the maximum length in the batch.
### Head
+ Same as FixLen with 1DCNN.

# Training
+ 3-fold participant CV.
+ 300 epochs, apply SWA after 15 epochs.
+ AdamW optimizer.
+ ArcMarginProduct is used, but only norm normalization of features and weight is applied since margin m=0.
+ Stochastic Depth, 0.1 or 0.5.
+ Label Smoothing Loss, epsilon=0.5.

# Data Augmentation
+ Randomly drop frames (p=0.3).
+ Augment hand position, size, and angle.
+ Sequence length input to 1DCNN-FixLen is between 64 and 128.

# CleanLab
+ CleanLab was used to remove approximately 5,000 scenes that were considered noise.
  + Calculated posterior probabilities with 21-participant-fold and used filter_by="both".
+ The model trained on the cleaned dataset increased LB in the stand-alone model, but not much when ensembling.
  + Is it because the data was too clean and the diversity of the models was reduced? I honestly don't know.
+ Also, the gap between CV and LB appeared, so I tried not to be too overconfident.
  + In the final submission, CleanLab was applied to only 2 of the 6 ensembles.

# Extra Experiments
I did some experiments, including some that weren't included in the final submission. The following two points can be made from this evaluation:
+ CleanLab is effective (+0.003) (compare A and B, D and E).
+ Large effect of ensembling FixedLen and VariableLen (+0.01) (A0+D).
+ I couldn't get the prize just by ensembling fixedlen.

|model|Length|head size|cleanlab|seed|Private|Public|
|:----|:----|:----|:----|:----|:----|:----|
|A0|fixed|deep|no|0|0.8661|0.7843|
|A1|fixed|deep|no|1|0.8682|0.7842|
|B0|fixed|deep|yes|5|0.8721|0.7848|
|B1|fixed|deep|yes|6|0.8702|0.7862|
|C0|fixed|shallow|no|5|0.8651|0.7794|
|C1|fixed|shallow|no|6|0.8665|0.7825|
|D|variable|deep|no|25|0.8653|0.7812|
|E|variable|deep|yes|35|0.8688|0.7840|
|F|variable|shallow|no|150|0.8647|0.7794|
|A0+A1|fixed|-|no|-|0.8722|0.7905|
|B0+B1|fixed|-|yes|-|0.8761|0.7935|
|A0+D|-|-|no|-|0.8766|0.7945|
|fixedlen only (A0+A1+B0+B1+C0+C1)|fixed|-|-|-|0.8774|0.7962|
|best sub(A0+B0+B1+C0+E+F)|-|-|-|-|0.8824|0.7999|
