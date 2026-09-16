# 3rd Place Solution

Competition: rfcx-species-audio-detection
Rank: #3
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220522

# 3rd Place Solution

**TLDR**

Our solution is a mean blend of 8 models trained on True positive labels of all recording ids in  train_tp.csv (given + hand-labeled labels) also from some recording ids in train_fp.csv (hand-labeled labels) with heavy augmentations. Additionally, some models are also trained on pseudo labels and a hand-labeled external dataset. We also post-processed the blended results by thresholding species 3.

From 308 submissions it is obvious that we have tested a lot of techniques and I will share more detailed information by category below.

## Data Preparation
I couldn't get a proper validation framework setup after trying out many techniques and decided at one point to start digging into the data and figured out that there are many unlabeled samples both in and out of the range of t_min and t_max labels given in train_tp.csv. In https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197735 it was mentioned that using hand-labeled species was allowed, so I started labeling the data manually and after labeling 100 recording ids I could already get a > 0.9 public lb score and pretty consistent local CV scores that somewhat correlates with the public lb. Naturally, I continued to label the entire train_tp.csv seeing that it has only around 1.3k recording ids.  Further labeling of train_fp.csv helped the score but only minimally so I stopped at one point. As I grew more familiar with the data I could label 300 recording ids in a day :), referring to pseudo labels helped a lot too. I also went through the train_tp.csv a few more rounds to make sure I have quality data. I used both spectrograms and listening strategy to analyze and label the data, some species are easy to spot with spectrograms and some are easier to spot by listening, and in some cases, both listening and visual inspection of the spectrograms can act as a multi verification technique to get more quality labels, especially when birds/frogs are very distant away from the recorder or there are strong noises like waterfall sounds. By labeling and analyzing the data I also figured out the kinds of sounds/noises that would appear and inspired me to try out a few augmentation methods which I will share below. Along with true positive labels, I also added noisy and non-noisy labels based on my confidence in the completeness of labels in a specific recording id. I am not a perfect labeler so I wanted to handle complete and non-complete labeled recording ids differently, which I will share below too.

I also removed some labels from train_tp.csv as I found some true positives suspicious, I didn't test not removing the labels before so not sure how much this helped.

Additionally, after finding out the paper from the organizers I searched for suitably licensed datasets with those species and found one dataset with species in this competition with a proper license. But there weren't any labels so I labeled it manually too with the same format as train_tp.csv. https://datadryad.org/stash/dataset/doi:10.5061/dryad.c0g2t . I reuploaded the dataset [HERE](https://www.kaggle.com/dicksonchin93/eleutherodactylus-frogs) with my manual labels.

I uploaded the extra labels as a dataset https://www.kaggle.com/dicksonchin93/extra-labels-for-rcfx-competition-data, feel free to use it and see if you can get a better single model score! mine was 0.970 on public lb
 
## Modeling / Data Pre-processing 

I used Mel Spectrograms with the following parameters: 32kHz sampling rate, a hop size of 716, a window size of 1366, and 224 or 128 Number of Mels. Tried a bunch of methods but plainly using 3 layers of standardized Mel Spectrograms works the best. The image dimensions were (num_mel_bins, 750). 

Using train_tp.csv to create folds will potentially leak some training data into your validation data so I treated the problem as a multilabel target and used [iterative-stratification](https://github.com/trent-b/iterative-stratification) to stratify the data into 5 partitions using unique recording ids and its multilabel targets. I had two different 5 fold partitions using different versions of the multi labels and used a mix of both in the final submission. 

I used multiple different audio duration during the competition and at different stages of the competition, the best duration varied in my implementation but in the end, I used 5 seconds of audio for training and prediction as the LWLWRAP score was better on both public lb and local validation. 

The 5-second audio was randomly sampled during training and in prediction time a 5-second sliding window was used with overlap and the max of predictions was used. How the 5-second audio is randomly sampled is considered to be an augmentation method in my opinion and so I will explain it in the heavy augmentations category below

###### Augmentations

- Random 5-second audio samples: 
a starting point was chosen randomly on values between reference t_mins and t_maxes obtained from 

```
def get_ref_tmin_tmax_and_species_ids(
    self, all_tp_events, label_column_key="species_id"
):
        all_tp_events["t_min_ref"] = all_tp_events["t_min"].apply(
           lambda x: max(x - (self.period / 2.0), 0)
        )
        def get_tmax_ref(row, period=self.period):
            tmin_x = row["t_min"]
            tmax_x = row["t_max"]
            tmax_ref = tmax_x - (period / 4.0)
            if tmax_ref < tmin_x:
                tmax_ref = (tmax_x - tmin_x) / 2.0 + tmin_x
            return tmax_ref
        all_tp_events["t_max_ref"] = all_tp_events[
            ["t_max", "t_min"]
        ].apply(get_tmax_ref, axis=1)
        t_min_maxes = all_tp_events[
            ["t_min_ref", "t_max_ref"]
        ].values.tolist()
        species_ids = all_tp_events[label_column_key].values.tolist()
        return t_min_maxes, species_ids

```
Labels were also assigned based on the chosen starting time and ending time with t_min and t_max labels.
- audio based pink noise
- audio based white noise
- reverberation
- time stretch
- use one of 16kHz or 48kHz sample rate data and resample it to 32kHz sample rate using randomly chosen resampling methods `['kaiser_best', 'kaiser_fast', 'fft', 'polyphase']`
- use different window types to compute spectrograms at train time `['flattop', 'hamming', ('kaiser', 4.0), 'blackman', 'hann'] `  , hann window is used at test and validation time
- masking out non labeled chunks of the audio with a 10% chance
- one of spectrogram [FMix](https://arxiv.org/abs/2002.12047) and audio based mixup with the max of labels instead of using the blend from the beta parameter
- [spec mix ](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Bouteillon_27_t2.pdf): 
only one strip was used for each axis, for the horizontal axis when the chosen frequency range to mask out completely covers a specific species minimum f_min and maximum f_max , that species label will be dropped. Specmix is also using the max of labels instead of using the blend from the beta parameter. The code below shows how I obtain the function that can output frequency axis spectrogram positions from frequency
```
def get_mel_scaled_hz_to_y_axis_func(fmin=0, fmax=16000, n_mels=128):
    hz_points = librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    hz_to_y_axis = interp1d(hz_points, np.arange(n_mels)[::-1])  
    # reversed because first index is at the top left in an image array
    return hz_to_y_axis
```
- bandpass noise 
- Water from [Freesound50k](https://zenodo.org/record/4060432#.YC6IJ3VKjCI) removing samples that have license to prevent derivative work
- Engine and Motor Sounds from [Freesound50k](https://zenodo.org/record/4060432#.YC6IJ3VKjCI) removing samples that have license to prevent derivative work
- Honk, Traffic and Horn sounds from [Freesound50k](https://zenodo.org/record/4060432#.YC6IJ3VKjCI) removing samples that have license to prevent derivative work
- Speech sounds from [Freesound50k](https://zenodo.org/record/4060432#.YC6IJ3VKjCI) removing samples that have license to prevent derivative work
- Bark sounds from [Freesound50k](https://zenodo.org/record/4060432#.YC6IJ3VKjCI) removing samples that have license to prevent derivative work


checkout recording_id b8d1e4865 to find dogs barking and some human speech :D

###### Architectures used
No SED just plain classifier models with GEM pooling for CNN based models
- Efficientnet-b7
- Efficientnet-b8
- HRNet w64
- deitbase224
- vit_large_patch16_224
- ecaresnet50
- 2x resnest50 from https://www.kaggle.com/meaninglesslives, checkout his writeup in a minimal notebook [HERE](https://www.kaggle.com/meaninglesslives/rfcx-minimal)!

###### Loss

The main loss strategy used for the final submission was using different loss function for samples which I am confident is complete in labels and samples which I am not confident is complete in labels. BCE was used for non-noisy/confident samples and a modified Lsoft loss was used on the noisy/non-confident. Lsoft loss was modified to be applied only to nonpositive samples, as I was confident in my manual labels. It looks like this 
```
def l_soft_on_negative_samples(y_pred, y_true, beta, eps = 1e-7):
    y_pred = torch.clamp(y_pred, eps, 1.0)

    # (1) dynamically update the targets based on the current state of the model:
    # bootstrapped target tensor
    # use predicted class proba directly to generate regression targets
    with torch.no_grad():
        negative_indexes = (y_true == 0).nonzero().squeeze(1)
        y_true_update = y_true
        y_true_update[negative_indexes[:, 0], negative_indexes[:, 1]] = (
            y_true_update[negative_indexes[:, 0], negative_indexes[:, 1]] * beta + 
            (1 - beta) * y_pred[negative_indexes[:, 0], negative_indexes[:, 1]]
        )
        
    # (2) compute loss as always
    loss = F.binary_cross_entropy(y_pred, y_true_update)
    return loss
```
This was inspired by the first placed winner in the Freesound competition https://github.com/lRomul/argus-freesound but I noticed that it doesn't make sense if it is used with mixup since audio will be mixed up anyways. So I also obtain the max of noisy binary labels so that noisy labels mixed with clean labels are considered to be noisy labels.

###### Pseudo Labels
I didn't get much boost from pseudo labels, maybe I did something wrong but nonetheless, it was used in some models. I used a 0.8 threshold for labels generated with 5-second windows and utilized the same window positions during training. Using raw predictions didn't help the model at all on lb.

###### Post processing
We set the species 3 labels to be 1 with a 0.95 threshold and it boosted the score slightly


###### Other stuff

- Early stopping of 20 epochs with a minimum learning rate of 9e-6 to start counting these 20 epochs
- Reduce learning rate on Plateau with a factor of 0.6 and start with a few warmup epochs, when LR is reduced the best model weights was loaded back again
- Catalyst was used

## Things that failed

- using models without pre-trained weights
- timeshift
- using species from the Cornell Competition that are confused with species in this competition as a distractor noise, for example, moudov is similar to species 15, reevir is similar to species 11, rebwoo is similar to species 6, bkbwar is similar to species 7, cacwre is similar to species 19, every is similar to species 17 and nrwswa is similar to species 20
- using plane sounds from Freesound50k data
- using PCEN, deltas or CQT
- [Random Power](https://www.kaggle.com/c/birdsong-recognition/discussion/183269)
- TTA with different window types
- Manifold mixup with resnest50
- using trainable Switchnorm as an initial layer replacing normal standardization
- using trainable Examplar norm as an initial layer replacing normal standardization
- Context Gating
- split audio intro three equal-length chunks and concat as 3 layer image
- lsep and [Assymetric loss ](https://arxiv.org/abs/2009.14119)
- using rain sounds from Freesound50k data
- using a fixed validation mask similar to how I used random training mask
- use [SIREN](https://vsitzmann.github.io/siren/) layer
- Tried to separate some confusing patterns as separate manual labels but didn't get the chance to test them


Hopefully, I didn't miss anything. Oh, we were holding off submitting a good model until @cpmpml came along :)
