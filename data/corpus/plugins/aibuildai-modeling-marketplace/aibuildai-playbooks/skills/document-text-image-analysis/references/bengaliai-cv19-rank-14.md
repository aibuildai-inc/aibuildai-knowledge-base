# 14th Place - Hacking Macro Recall - Chris Writeup

Competition: bengaliai-cv19
Rank: #14
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136021

This is my first computer vision gold medal. I'm very excited!

Thank you Kaggle and Bengali.AI for hosting a fun comp. Thank you teammates Bojan, Shai, Yasin, Jahmed ( @tunguz @sgalib @mykttu @jasemahmed ). I had a blast working with you guys! Thank you Nvidia for providing GPU compute. Below are my contributions to our team's solution. The rest of the team will share more.

# Competition Metric Explained

This competition's metric is macro recall. That means you compute the recall of each class individually and average them. Most importantly `recall = found / exist`. There is no penalty for making a false positive! Making more positive predictions for one class can only increase that class' recall never decrease!

Below is a paradoxical example. Imagine that your are predicting one of seven consonant diacritic classes and your CNN outputs the following probabilities:

## Softmax = [0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0]

## Question: Do you predict class 1 or class 2?

Let's say that there are 1000 samples in class 1 and 100 samples in class 2. Then if you predict class 1 you have a 70% chance of increasing your class 1 recall by 1/1000. If you predict class 2 then you have a 30% chance of increasing your class 2 recall by 1/100.

Therefore your expected macro recall increase if you predict class 1 is `1e-4 = 0.70 * 1/1000 * 1/7`. And your expected macro recall increase if you predict class 2 is `4e-4 = 0.30 * 1/100 * 1/7`. Therefore you predict class 2.

## Answer: You predict class 2 not class 1
By adjusting your predictions in this fashion you can gain a massive 0.0027 public LB increase and 0.0241 private LB increase!




### Code
Try the following post process on your model to see how much it increases your public and private LB. 
    
    preds = model.predict(X_test)
    p0 = np.argmax(preds[0],axis=1)
    p1 = np.argmax(preds[1],axis=1)
    p2 = np.argmax(preds[2],axis=1)

    EXP = -1.2

    s = pd.Series(p0)
    vc = s.value_counts().sort_index()
    df = pd.DataFrame({'a':np.arange(168),'b':np.ones(168)})
    df.b = df.a.map(vc)
    df.fillna(df.b.min(),inplace=True)
    mat1 = np.diag(df.b.astype('float32')**EXP)

    s = pd.Series(p1)
    vc = s.value_counts().sort_index()
    df = pd.DataFrame({'a':np.arange(11),'b':np.ones(11)})
    df.b = df.a.map(vc)
    df.fillna(df.b.min(),inplace=True)
    mat2 = np.diag(df.b.astype('float32')**EXP)

    s = pd.Series(p2)
    vc = s.value_counts().sort_index()
    df = pd.DataFrame({'a':np.arange(7),'b':np.ones(7)})
    df.b = df.a.map(vc)
    df.fillna(df.b.min(),inplace=True)
    mat3 = np.diag(df.b.astype('float32')**EXP)

    p0 = np.argmax( preds[0].dot(mat1), axis=1)
    p1 = np.argmax( preds[1].dot(mat2), axis=1)
    p2 = np.argmax( preds[2].dot(mat3), axis=1)

# Chris Model
Our team's solution is an ensemble. For my model I trained a 128x256 efficientNetB6 150 epochs with CAM CutMix in addition to basic rotation, scale, shift, cutout, and cutmix. Training took 24 hours on four Nvidia V100 GPUs. It's CV is 0.9967, public LB 0.9916, and private LB 0.9544. 
  
CAM CutMix is where you find the class activation maps of the images and then remove the most important parts of the image and replace them with another image. This challenges your CNN and helps it generalize.

## CAM Maps
First you train one model to produce CAM maps. (First model was 256x256 efficientNetB6). Next you use CAM maps to train a second model. (Second model was 128x256 efficientNetB6). Here are some CAM maps: (More pictures [here][1]).






## CAM CutMix

CutMix is a combination of two images. The first image is displayed as yellow below to help us visualize it. First one component either root, vowel, consonant is randomly selected. Next a random percentage from 15% to 25% is selected. Next that percentage of the first image is removed using the chosen component type's CAM map. Finally the same region from a second randomly selected image is inserted. The second image is displayed as blue below. 50% of images use CAM CutMix and 50% use regular CutMix. (CAM CutMix increased CV and LB by 0.001 over regular CutMix).



[1]: https://www.kaggle.com/c/bengaliai-cv19/discussion/136025
