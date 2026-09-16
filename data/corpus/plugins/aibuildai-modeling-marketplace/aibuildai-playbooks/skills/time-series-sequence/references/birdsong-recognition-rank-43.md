# 43rd place solution

Competition: birdsong-recognition
Rank: #43
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183240

Big thank you to HOSTKEY for allowing me to use a machine with 2x1080Ti as a grant. You can check them out here: https://www.hostkey.com/gpu-servers#/ HOSTKEY servers are cheaper than AWS and Google Cloud, and they are offering pre-orders for servers with RTX 3080's on them.

The goal of this competition is to predict the species of bird in a soundscape recording, given non-soundscape recordings. Basically, they give you a 5 second clip recorded in a forest, and you have to say what birds they are. There are 264 species in total to predict.

The difficulties in this competition come from: 
1. The training data is from recordings of any birdwatchers - anyone can record and upload, which means sometimes it is recorded just on a smartphone, there is speech in it, etc. However, the testing data is recorded from boxes strapped to trees, recording for 10 minutes at a time.
2. The training data is of variable length, from seconds to minutes. The data is recorded at different sample rates, at different volumes, in different locations. The test data comes from 3 unknown sites.
3. Some species of birds make different birdcalls even though they are the same species. There are regional dialects of birdcalls. And some calls can vary (alarm call, mating call, etc.)
4. The training clips can have different birds in the background, or they have long periods of no birds.
5. You can have false positives through other animals like chipmunks, cicadas, flies, cars driving nearby, etc.

It is evident that the training data is extremely different from the testing data. I noticed that an improvement in my validation loss on a 20% holdout set from the training set yielded weaker leaderboard results on the hidden test set.

Then, I decided to train for lots more epochs, past the optimum for my validation loss. This ended up getting better on the leaderboard. So I concluded that training more epochs = better score, even if I considered it locally overfit.

In order to address Difficulty #1, I randomly augment my training clips. I add pink noise with varying volumes, and random soundscape recordings (up to 3 with different volumes). I also randomly applied a Butterworth filter (randomly lowpass, highpass, bandpass, bandstop) with random cutoffs. I also used Cutout augmentation, which randomly replaces an area of the clip with noise pixels. I also randomly use ColorJitter to change saturation, hue, brightness, and contrast. ***You can see I use the word "random" so much -- I really wanted to make sure my submission was very robust.*** Much of these ideas are inspired from previous Birdclef solution: http://ceur-ws.org/Vol-2125/paper_140.pdf

In order to address Difficulty #2, I randomly sample 5 second clips from the training clips. If the clip is less than 5 seconds, then add 0's to the start/end of the clip randomly.

I hoped the model could cope with Difficulty #3 by itself.

In order to address Difficulty #4's long periods of no birds, I removed contiguous stretches >= 4 seconds in my training clips where the absolute signal amplitude doesn't exceed the 99.9th quantile. This effectively removed long contiguous seconds of silence, which allows my model to focus more on the birdcalls and less on the absence of birds. I did not use "secondary_labels" for different birds in a single clip because I found it empty or inconsistent in many clips.

In order to address Difficulty #5, I also added examples of chipmunks/insects as background noise to tell my model that it is an absence of birds.

All models converted the 5 second training clip into a Melspectrogram, which is a 2D "picture" that represents what the sound looks like. Some models then used Power_to_Db function to convert the power spectrogram to decibel units; this is called Log-Melspectrogram. Other models used PCEN which is a novel transformation shown to outperform Log-Melspectrograms (http://www.justinsalamon.com/uploads/4/3/9/4/4394963/lostanlen_pcen_spl2018.pdf).

I trained one Efficientnet-B1, two Efficientnet-B2, one Efficientnet-B3 (https://arxiv.org/abs/1905.11946), one Resnest50 (https://arxiv.org/abs/2004.08955), two Inceptionv4 (https://arxiv.org/abs/1602.07261), and one SE-Resnext (https://arxiv.org/abs/1709.01507). I also trained an Efficientnet-B5 and more Resnest50/101, but it would not fit into the runtime to use these models. I also trained a 1D Convolutional Net but it wasn't strong enough. All models were initialized with pretrained imagenet weights. Some models have mixup and some have label smoothing for additional diversity.

In order to fit all of these models in the runtime, I converted all models into ONNX format which brought significant speedup.

Instead of averaging the predictions, I found that squaring the predictions, taking the mean, and then taking square root was better (follows from Lasseck's findings in previous Birdclef competition). I early stopped the models through intuition and Leaderboard feedback.

I trained on full 100% training data blindly, using BinaryCrossEntropyWithLogits as the loss. All models have different Melspectrogram parameters. I found that using high `fmin` parameter was good to get rid of some noisiness (effectively simulates a "zoom" into the birdcalls). Different image sizes and `n_mels` were also used. Different model heads were used (with different dropouts and number of Linear layers)

Special thanks to the strong Japanese Kaggle contributors (Tawara and Hidehisa Arai) and thank you to thesoundofai.slack.com for inspiration and tips.

Things that worked:
- Cutting out silence from training data
- Overlaying noise onto training data
- Data augmentation methods (noise injection, Cutout, ColorJitter, etc.)
- Squaring predictions, averaging, then Square Rooting
- Ensemble with different parameters

Things that didn't work:
- ArcFace Loss
- Custom F1 Row-wise Micro Loss
- MultiLabelSoftMarginLoss
- Freesound2019 Winning CNN solution architecture
- Removing the top k losses from each batch, assuming some clips are still noisy/incorrect in training data
- Using Freesound non-bird audio external dataset
- Using NIPS 2013 Bird identification external dataset

Things I didn't try:
- Adding in MFCC information or other numeric features
- Using external Xenocanto data
- PANN/pretrained "audionet" models
