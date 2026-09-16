# 39th place solution [top1 at public]

Competition: birdsong-recognition
Rank: #39
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183258

## Summary
1. Noise is the key
2. Test might be recorded with 16 kHz sampling rate
3. Sequence-wise predictions transformed into global and 5s chunk predictions with logsumexp pooling (training on 20-40s segments, inference on full files)
4. Multi-head self-attention applied to entire sequences

Congratulations to all participants and thanks to organizers for making this competition possible. Also, I would like to express my gratitude to my teammates for working together with me on this challenge. Below I will share some ideas used in our solution. Since I have been using quite a different approach from most of people in this competition, I decided to prepare a write-up regarding my part. 

## Look into data
It's probably the most important thing, especially for a competition like this.
**Test might be recorded with 16 kHz sampling rate**. Look at the example of test data:

There is a frequency gap above 8 kHz that could suggest either that the data is recorded with 16 kHz sampling rate and then up-sampled to 32 kHz or that there is some noise filter used (which would be quite unlikely). Meanwhile below 1 kHz the noise it too high to recognize anything. Therefore, I've chosen **1-8 kHz range** for mel spectrograms using 128 mels. Based on CV drop, frequencies above 8 kHz might be important, but they are not present in test data, and high CV may be misleading. Also models may learn features not present in test if frequencies above 8 kHz are used.

**Noise is the key**. The test data depicted above is quite noisy and the overall level of signal is weaker than one in train (meanwhile while noise+signal is similar in train and test). So several things were used: adding white noise and taking test noise extracted by @theoviel and posted [here](https://www.kaggle.com/theoviel/bird-backgrounds). In the second case the train signal is multiplied by an exponential random variable with lambda 0.25 limited at [0.1,1] and added to a randomly selected test noise chunk from concatenated noise array.

The produced train example looks quite similar (lower image) to test examples. Meanwhile, the original train data (upper image) has many features that could not be recognized at a high level of noise, and facilitates creation of a model that is good at CV but bad at test. Even training for several epochs with noise substantially improves the quality of the model, so top 5 predictions on test examples start making sense:


## Model
I used a quite different approach from most of participants, which I schematically depict below:

Instead of working with 5s segments, I worked with sequences: 20 and 40s for training and entire audio for inference. I collapse the frequency domain into dim of size 1 and then consider the produced tensor as a sequence and apply multi-head self-attention blocks to it, like in transformers. The produced output with stride of ~0.3s is merged with logsumexp pooling to produce the prediction for the entire audio segment or 5s intervals when run prediction on the test. During training the loss is computed based on global labels. I attached several examples below showing the prediction for top5 classes over time of my best model for first 40s of both test examples.

The spikes coincide with birdcalls, and if organizers provided more time resolved examples, sufficient at least to properly initialize the model, this method would be performing even better.
Training on 20 and 40s intervals is chosen to mitigate the possibility of having nocall in a sampled train chunk. In addition, consideration of an entire sequence during inference utilizes global attention, so the model is capable to incorporate knowledge about noise characteristics and different calls of the same bird when generate the predictions. Also, it naturally produces global predictions used for site_3. 
The basic kernel showing training on 5s chunks and reaching 0.65 CV in 16 epochs is posted [here](https://www.kaggle.com/iafoss/cornell-birdcall). It provides the details of implementation of the above approach.
The performance of the best single model is **0.622/0.588** private/public LB. Ensemble of my models with more traditional models trained by my teammates (prediction based on 5s chunks with a number of additional tricks) boosted our public LB to 0.628 within last several days but unfortunately only slightly improved private LB, giving 0.622.

**Additional details:**
Backbone: ResNeXt50
Loss: Focal loss, corrected to be suitable for soft labels
Augmentation: MixUp, white and test noise, stretch, temporal dropout.
External data: images beyond 100 examples
Use secondary labels with 0.1 contribution

**Postprocessing**: I have been using quite a complex pipeline finetuned on test examples: in this competition I made only ~15 subs. First, I generate global predictions above the threshold, based on logsumexp of the predicted sequence, and selecte top3 or top4 of them if their number is large. Next, I compute predictions for 5s chunks (using logsumexp of parts of the predicted sequence), selected ones that are above a particular threshold in comparison with their average value, and dropped all of predictions not listed in global ones. Finally, as suggested by my teammate @kirillshipitsyn , if both neighboring chunks have the same bird predicted I added this bird as a prediction, which gave 0.001+ boost. 

Some words about validation. I mostly considered test examples as a way to assess how good it the model. In my nearly first attempt I got ~0.80 CV (computed for the best threshold based on 4 fold train/val split) when trained on 20s chunks and ~0.83 CV when continued training on 40s chunks. However, when I checked the performance of the model on test examples, I realized that it predicts nearly nothing. Moreover, even top predictions are quite different from that should be. So I started adding such tricks as noise and 1-8kHz frequency range, which reduce CV but improve the model performance at test examples and LB. A good way to perform CV in this competition would be generating a val set based on train data with adding noise and excluding frequencies beyond 8 kHz, to make sure that it is as similar as possible to test examples. But I realize it nearly at the end of the competition. If I joined it not just 2-3 weeks before the deadline, probably, I could have more time to explore and fully handle the above ideas, and hopefully get better score at LB.
