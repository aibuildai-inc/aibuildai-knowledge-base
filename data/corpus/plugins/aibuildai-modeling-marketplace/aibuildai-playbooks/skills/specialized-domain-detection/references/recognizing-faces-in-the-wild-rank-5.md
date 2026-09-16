# 5th place solution review | quick notes | private 0.917

Competition: recognizing-faces-in-the-wild
Rank: #5
Source: https://www.kaggle.com/c/recognizing-faces-in-the-wild/discussion/103543#latest-596448

In short, my result has been achieved by my submissions blending with public available submissions from @hsinwenchang , @shivamsarawagi , @pasquale , @arjunrao2000 . Thank you guys!.

My path:

**Step 1 - Kinships embedding**
I have started from not very deep ResNet and, what can be more interesting, from metric loss! I Just divided training set into pairs of kinships and let my PC to search for a hyper space in which kinships are grouped together while non-kinships are spread. Not very good solution, because huge amount of the same faces belongs to different classes. However, I got 0.698. Experiments with different arches and data augmentation did not change score much.  

**Step 2 - Dlib's face descriptor**
Well, what is going on in the discussion thread? Seems face descriptors are fired (special thanks to @CVxTz for discussion thread). But almost all kernels refers to VGGFace work. Why not to check something different? There is one great work described [here](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html). So, I have tried. Raw Dlib's ([@DavisKing](https://github.com/davisking)) face descriptor allowed to get 0.754.

**Step 3 - Back to the training data**
My next idea was to train net with cross entropy loss function that takes as the net input a stack of two images. First experiments allowed to get 0.705 PLB AUC. Many hours later, this strategy allowed to get 0.760. What did make this improvement? Switching from 6 channels input (2 RGB photos) to 2 channels input (all photos converted to greyscale with centering and normalization). Adding horizontal flip for data augmentation. Training with cross validation. Details can be found [here on github](https://github.com/pi-null-mezon/Kaggle/tree/master/Kinship/Crossenthropy/CVLearner).      

**Step 4 - Cross entropy head for face descriptor**
I have always want to try to train net on top of the features produced by the other net. It was great opportunity to try this kind of strategy. I have started from training simple network (two fully connected layers) on top of Dlib's face descriptor features. Pipeline of data preparation has been prepared at previous steps, I have to simply add faces descriptions evaluation. Two input photos converted into two feature-vectors, then this vectors goes to mini batch along with 'kinship' or 'non-kinship' label. And, almost from the start I have got 0.803. Wow! For me it was great forward step. I have start to experiment with features combination and eventually come to training model pipeline that produced single network with PLB AUC 0.820+-0.015! Main progress at this point has been achieved by means of train mini batch structure adjustment. For the instance mini batch contained from 40 to 50 anchor persons and each anchor 25 kinships and 25 non-kinships. All [details](https://github.com/pi-null-mezon/Kaggle/blob/b28601c9a3ee6bab0540f9f4b9c2e870fa37bea6/Kinship/Crossenthropy/Headlearner/main.cpp#L263) on github. It was great. Moreover blending submissions of several networks trained on different cross validation folds allowed to get score around 0.857.

**Step 5 - Blending with public submissions**
It was interesting to blend my submissions with something different. Thanks to ['blend-of-smiles'](https://www.kaggle.com/vaishvik25/blend-of-smiles) kernel it was easy to make it (i am not very familiar with python tools, so for me it is was helpfull, thank you @vaishvik25 ). Almost immediately, when I have first visually compared distributions (kdeplot from seaborn) of 'is_related' values for my submission and public submissions I have noticed that they were very different. And it was good sign because at the same time PLB AUC-s were almost same. 


Eventually, blending experiments with different weights for submissions allowed to get PLB score 0.906.

**Step 6 - More different submissions for blending**
Where else we can get original submissions? Basically, looking on competition progress we can see that to produce good submission we need to use good face description model. Such as 'vgg_face' or Dlib's face descriptor. Of course we can search such models on the web. But only two weeks until competition ends, and almost no free time to debug all of possible issues. We need some kind of more trustworthy and reliable solution. Why not to train our own face description model? Maybe several models! And then, the heads for them! All training pipelines are ready. Just need to get somewhere huge face photo dataset. Luckily [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) exists. Download. Train face description model (eventually I have trained 7 models). Train kinship recognition models. Generate submissions. Blend. My final PLB AUC was 0.912 and it was 20th place. Not very great progress from 0.906? But luckily on the private leader board my submission turned out to be less over fitted than others.

Thank you!
