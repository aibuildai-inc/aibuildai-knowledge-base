# 15th Place Solution

Competition: child-mind-institute-detect-sleep-states
Rank: #15
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/460177

First of all, I'd like to thank kaggle team, host for hosting this competition and thank many other participants to share codes, ideas and datasets. Special thanks to my teammates ([furu](https://www.kaggle.com/kunihikofurugori), [mizoo](https://www.kaggle.com/yukio0201), [isamu](https://www.kaggle.com/yamsam), [tereka](https://www.kaggle.com/tereka)) who worked together on the competition and [213tubo](https://www.kaggle.com/tubotubo) for sharing such a great [pipeline](https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states).

### Overview

Our final submission is ensemble of 55 models in total, including eleven 5fold models.

Our solution has four main parts. 
They are furu, mizoo, kaerururu part, isamu part, tereka part, and postprocess part.

### furu, mizoo, kaerururu part

Our pipeline is based on 213tubo's one.
The architecture is various Feature Extractor -> Unet Encoder -> Unet1d Decoder.
Onset cv best model and wakeup cv best model are `saved separately`.

- Feature Engineering
    - normalize anglez, enmo with mean, std
    - anglez, enmo, hour feature (sin, cos)

- Feature Extractor
    - LSTMFeatureExtractor
    - CNNSpectrogram
    - GRUFeatureExtractor
- Unet Encoder
    - various pretrained weight
        - resnet18
        - resnet34
        - inceptionv4
- Unet1d Decoder
- various duration
    - 17280 (24h)
    - 5760 (8h)

### isamu part

Isamu has two model architectures. The one is MultiResidualBiGRU model based on [4th place solution of parkinson competition](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/416410) and the other is Wavenet based.

- Feature Engineering
    - normalize anglez, enmo with mean, std
    - stack sequences down sampled with various way (min, max, mean, std, median) 

### tereka part

Tereka has two model architectures. The one is Unet based and the other is Transformer based.

- Feature Engineering
    - normalize anglez, enmo with mean, std
    - anglez, enmo, diff (anglez, enmo), hour feature (sin, cos)

- Use a moving average  (window_size=12) of the model output.

### post processing

- Unwear detection
    - Both rule base (focus on daily cycle) and random forest model base are worked.
    - One of the final submissions was with rule base and the other was with random forest model base.
- When predicted value step=0, set step value to +1
- Boost candidates within 24 hours
    - Considering the periodicity of event being one within 24 hours, candidates with the greatest probability within 24 hours were boosted

### What Worked
- Remove noisy 8 sample in train and valid phase
- Add fulltrain
- The lower threshold, the better score
   - The lower threshold, the higher Probability of submission scoring error :(
- Exponential Moving Average
- Label Smoothing
- combine bce loss with dice loss(only apply peak)


### What Didn’t Work

- Larger model backbone
- positional encoding
- date embeddings
- 2nd stage NN model
- layer lr tunning
- Transformer decoder
- The 0:3000 steps of onset predicted value array and the -3000: steps of wakeup predict array are set to 0.

### Comparison of CV, LB, and Private for the final submissions

| Name | CV | LB | Private |
| ---- | ---- | ---- | ---- |
| Submission 1 (cv best)  | 0.8125 | 0.773 | 0.823 |
| Submission 2 (lb best)  | - | 0.778 | 0.822 |

### Comparison of CV, LB, and Private with and without post-processing

| Name | CV | LB | Private |
| ---- | ---- | ---- | ---- |
| With post-processing  |  0.8125 | 0.773 | 0.823(+0.011) |
| Without post-processing  | - | 0.767 | 0.812 |

### Best scores for each single model

| Name | Model | CV | LB | Private |
| ---- | ---- | ---- | ---- | ---- |
| furu model  |  GRUINITFeatureExtractor-UNet1DDecoder  | 0.7689665 |  0.746  |  0.8  |
| kaerururu model1  |  GRUFeatureExtractor-UNet1DDecoder  | 0.743 | - | - |
| kaerururu model2  |  GRUFeatureExtractor-UNet1DDecoder  | 0.733 | - | - |
| mizoo model  |  LSTMFeatureExtractor-UNet1DDecoder  | 0.7508 | 0.736 | 0.784 |
| Isamu model1 |  wavenet_lstm  | 0.773 | - | - | 
| Isamu model2 |  1d-gru  | 0.775 | - | - | 
| tereka model1 |  unet base  | 0.764 | 0.752 | 0.803 |
| tereka model2 |  transformer base  | 0.773 | 0.729 | 0.797 |


### Important Citations
- 213 tubo's pipeline
    - https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states
- parkinson 4th place soludion
    - https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/416410
