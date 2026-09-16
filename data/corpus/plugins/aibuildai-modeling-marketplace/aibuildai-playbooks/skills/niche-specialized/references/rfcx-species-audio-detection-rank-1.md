# 1st place solution

Competition: rfcx-species-audio-detection
Rank: #1
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220563

Thanks to Kaggle and hosts for this very interesting competition with a tricky setup. This has been as always a great collaborative effort and please also give your upvotes to @christofhenkel and @ilu000. In the following, we want to give a rough overview of our winning solution.

### TLDR
Our solution is an ensemble of several CNNs, which take a mel spectrogram representation of the recording as input and predict on recording level using “weak labels” or on a more granular time level using “hard labels”. Key in our modeling is masking as part of the loss function to only account for provided annotations. In order to account for the large amount of missing annotations and the inconsistent way how train and test data was labeled we apply a sophisticated scaling of model predictions.

### Data setup & CV
As most participants know, the training data was substantially differently labeled compared to the test data and the training labels were sparse. Hence, it was really tricky, nearly impossible to get a proper validation setup going. We tried quite a few things, such as treating all top 3 predicted labels as TPs when calculating the LWLRAP (because we know that on average a recording has 3 TPs), or calculating AUC only on segments where we know the labels (masked AUC), but in the end there was no good correlation that we could find to the public LB. This meant that we had to fully rely on public LB as feedback for choosing our models and submissions. Thankfully, it was a random split from the full test population, but everything else would not have made much sense anyways most likely.

### Models
Its worthy to note that for most models we performed also the mel spec transformation and augmentations like mixup or coarse dropout on GPU using the implementation that can be found under torchlibrosa ([https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/stft.py](https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/stft.py)).

Our final models incorporate both hard and weak label models as explained next.

#### Hard label models
We refer to hard labels as labels that have hard time boundaries inside the recordings. Our hard label models were trained on the provided TPs (target = 1) and FPs (target = 0) labels with time aware loss evaluation. We used a log-spectrogram tensor of variable time length as input to an EfficientNet backbone and restricted the pooling layer to only mean pool over the frequency axis. After pooling, the output has 24 channels for each species and a time dimension. 

We then map the time axis from the model to the time labels from the TPs and FPs and evaluate the BCE loss **only** for the parts with provided labels. For all other segments (which is actually the majority) the loss is ignored, as we have no prior knowledge about the presence or absence of species there. In the figure below we show how a masked label looks like: yellow means target=1, green is target=0 and purple is ignored.


For some models we added hand labeled parts of the train set but saw diminishing returns when labeling species that were missed by the TP/FP detector, which makes us wonder how the test labeling was done. Also, we wonder where the cut was made for background songs (e.g. species 2 had some calls in the background of several recordings, but the parts were labeled as FP). Most notably, adding TP labels for species 18 gave a substantial boost to LB score, and we believe that adding some hand labels to the mix of models in the blend helped with diversity and generalization. 

For some models, similar to other top performing teams, we trained a second stage in which we replaced the masked part of the label with pseudo predictions of the first stage, but downweighted with factor 0.5. The main difference here to other teams is that we scaled the pseudo predictions in the same way we scale test predictions.

As augmentation we used mixup with lambda=3, SpecAugment and gaussian noise.

#### Weak label models
The models in this part of the blend are based on weak label models. The input is the log-spectrogram of the full 60 seconds of an audio recording including all the labels for that clip. So it directly fits on the format where the final predictions need to be made. Due to missing labels, just fitting on the known TPs does not work too well as we incorporate wrong labels by nature. Also we cannot use the FPs, because even though an FP might be present in one part of the recording, does not mean there might not be a TP at another position.

Hence, the models fit here include pseudo labels from our hard label models (see above) as well as some partial hand labels. For the pseudo labels, we take the raw output from the hard label models, but scale them to our expected true distribution (see post processing). For the hand labels, we only pick the TPs as well as FPs that span over a 60second period so that we are sure the species is not part of that recording. In loss, we weight the pseudos between 0.3-0.5 and the original labels and hand labels as 1.

If we would just fit on the raw pseudo outputs, we would not learn anything new, so we employ concepts from noisy-student models. That means we utilize not only simple augmentations and mixup, but also randomly sample pseudo labels for each recording each time we train on it based on a pool of stage 1 hard label models. So for example, you fit 10 hard label models, and then randomly sample one each time in the dataloader. This introduces randomness and further boosts on top of the stage 1 models.

Additionally, we fit several backbones (efnetb0, efnetb3, seresnext26, mobilenetv2_120d) where each is trained on the full data (no folds) with several seeds. In the end this part of the blend is a bag of around 120 models, where some also have additional TTA (horizontal flip).

### How we are blending
We are blending different model types described above as depicted by the following graphic:


### Post processing
We noticed that the test distribution of the target labels is substantially different to the provided train labels. Due to this fact, the models assume an unreasonable low or high probability when they are uncertain (Chris already has started a great thread about it [here] (https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220389)). To tackle this, we used several a priori information from the test distribution and scaled our predictions accordingly: by probing the public leaderboard we extracted a test label distribution which was aligning well with a previous research paper from the hosts. With additional prior knowledge about the average number of labels per row (3) -- also confirmed by LB probing, as well as the research paper -- we applied either a linear (species_probas *= factor) or a power scaling (species_probas **= factor) per species to our predicted probabilities to match the top3 predictions distribution (orange) with the previously mentioned estimated test distribution (blue). But we didn’t stop there, as we know that the number of labels per row is not always 3 but can be as low as 1 or as high as 8 (stated in the paper). Based on the sum of our probas in each row, we estimated the most likely topX (with a minimum count of 1) distribution (green) of the test set, and optimized the scaling factors by minimizing the total sum of the errors. 


### What did not work
I think in the end quite a few things we tried ended up in the blend fostering the diversity in it. But naturally, there are also many different things that did not work, after all we ran close to 2,000 experiments throughout the course of this competition. One noteworthy thing we tried was object detection based on the bounding boxes we had available in training. It worked reasonably well on simple CV setting reaching >0.7 LWLRAP on full 60 second recordings, but we never continued to work on it on smaller crops or other settings.

We explored quite some architectures in the hope to improve our ensemble. So we tried models that work on the raw wave like Res1DNet or the just released wav2vec. But none did sufficiently well.

Thanks for reading. Questions are very welcome. 
Christof, Pascal & Philipp
