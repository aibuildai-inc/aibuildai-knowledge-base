# 40th Place Solution - Improving prediction with FFT-based data cleaning

Competition: child-mind-institute-detect-sleep-states
Rank: #40
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459648

First of all, I would like to thank [@tubotubo](https://www.kaggle.com/tubotubo) for sharing your high-quality code. I am going to share my solution.
This is my first post on a solution, so I apologize if I'm being rude in any way. Although the content is poor, I hope it will be of help to you.

### 1. Data preprocessing
- Null Value Removal: Removed rows with null timestamp values in the dataset to prevent misinterpretation of the analysis and reduce uncertainty in model training.
- Event count consistency check: Filtered out unmatched data at the beginning ('onset') and end ('wakeup') of events, increasing data integrity and analysis reliability.


### 2. Feature Engineering
Cleaning features using FFT: Referring to [cmi-sleep-detection-fast-fourier-transformation](https://www.kaggle.com/code/jjinho/cmi-sleep-detection-fast-fourier-transformation), data was cleaned using different thresholds (98.75, 99.0, 99.5, 99.75, 99.9) for enmo and anglez. This generated FFT-based features (e.g. fft_9875).


### 3. Model
For cross-validation, I use a 5-fold GroupKFold.
I built seven different models based on [@tubotubo's code](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/452940) , each using a different feature set and network architecture (LSTM, GRU, UNet, Transformer, etc.).

| model | CV | Pubic (5-fold) | Private (5-fold)|
| --- | --- | --- | --- |
| FeatureExtractor(LSTM(dim=64)+GRU(dim=64)) + UNet + UNet1DDecoder with no fft feature| 0.74373|0.746|0.791|
| FeatureExtractor(LSTM(dim=64)+GRU(dim=64)) + UNet + UNet1DDecoder with fft_9900, fft_9950, fft_9975, fft_9990| **0.75250**|0.727|0.791|
| FeatureExtractor(LSTM(dim=64)+GRU(dim=64)) + UNet + UNet1DDecoder with fft_9875| 0.76012|0.747|0.795|
| FeatureExtractor(TransformerFeatureExtractor(dim=64)) + UNet + UNet1DDecoder with fft_9875| 0.74572| | |
| FeatureExtractor(LSTM(dim=128)+GRU(dim=128)) + UNet + UNet1DDecoder with fft_9875| **0.76400**|0.739|0.801|
| FeatureExtractor(LSTM(dim=128)+GRU(dim=128)) + UNet + UNet1DDecoder with fft_9900| 0.76208|0.735|0.795|
| FeatureExtractor(CNNSpectrogram(dim=128)) + UNet + UNet1DDecoder with fft_9875| 0.74933| | |



Cleaning features using FFT had a big effect on CV, but when I opened the lid, it didn't seem to have a big effect on the private score.
By increasing the number of dimensions during feature extraction from 64 to 128, the accuracy of CV and private score was significantly improved.


### 4. Model Ensemble
Two-step ensemble approach: 
In the first step, predictions from seven models were ensembled using three different methods: Optuna, Nelder-Mead, and Hill Climbing.
Next, in the second step, these three types of ensemble results were further ensembled using Optuna to obtain the final results.
This approach yielded good results in both cross-validation and private scores. The first and second stages of Private were both 0.802, but the second stage was slightly better, but it may not have been necessary to go this far.

### first stage
| Ensemble method | CV | Pubic (5-fold) | Private (5-fold)|
| --- | --- | --- | --- |
| optuna| 0.79038|0.746|0.802|
| nelder-mead| 0.78951| | |
| optuna| 0.78801 | | |

### second stage
| Ensemble method | CV | Pubic (5-fold) | Private (5-fold)|
| --- | --- | --- | --- |
| optuna| 0.79049|0.746|0.802|


### 5. Approaches that didn't work
- Applying Focal Loss: The cross-validation score improved when using Focal Loss, but an error occurred during submission.
- Applying asymmetric Gaussian and exponentially decaying distributions: I applied asymmetrically Gaussian and exponentially decaying distributions to the labels, but these approaches did not contribute to improved performance.
- Additional feature engineering: I tried additional features such as moving average, standard deviation, difference signal, cumulative sum, and autocorrelation, but these also did not improve performance.
- Using DeBERTa-v3-small: I used DeBERTa-v3-small for feature extraction, but learning did not proceed well and I did not get the expected results.



Finally, thank you for organizing the competition. It was a very challenging competition and I learned a lot.

Moreover, this is my second silver medal. I want to continue working hard to become a kaggle master. Thank you everyone for letting me learn so much. I will continue to do my best.
