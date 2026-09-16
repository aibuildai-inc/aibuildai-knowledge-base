# 29th Place Solution and Initial Solution

Competition: child-mind-institute-detect-sleep-states
Rank: #29
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/460021

## 29th Solution Overview:
### Model Architecture (from tubo's GitHub Repository):
- 2DCNN Feature Extractor -> Unet -> Unet Decoder
- LSTM Feature Extractor -> 1DCNN -> LSTM Decoder

### Input Features:
- enmo, anglez, anglez_diff, hour, weekday, and a noise flag.
  - Noise is flagged if identical (enmo + anglez) occur within a 5-minute span in the same series_id.
  - EDA indicated that the day before a holiday typically shows later onset times, prompting the inclusion of weekday and holiday features.
- Input steps of 5760 and 11520 were used, with four models based on ensemble model and input step pairs.

### Target Size:

- The target size is identical to the input size.

### Training Targets:

- Events with Gaussian soft labels (2 channels)
- Sleep flag (1 channel)

### Inference Process:

- Chunks are created by sliding the input_step/4.
  - For example, in the 5760 model, chunk steps begin from 0, 1440, 2880, etc.
- All chunks are predicted and averaged by step.
- In CNN-based models, edge predictions are trimmed by 12.5%.

### Postprocessing:

- Use only events prediction (sleep flag is not used).
- Utilizes find_peaks from SciPy signals, and applies a weighted average to steps near detected signals based on the prediction scores.
  - For instance, if the candidate step is 5000, the weighted average of steps 4998-5002 is calculated.

<hr>

Final solution is just a small modification from tubo's approach, so let me share my initial solution. (I took most of my competition time in this approach...)

## Initial Solution (LB 0.677):

### Based on a two-stage model:
- Stage 1: Chunk Detection by binary classification 
  - Segmenting the dataset into 1-hour chunks with weak labels to identify useful segments. Label 1 is added if an event occurs within a chunk.
- Stage 2: L1 Regression 
  - Pinpointing the event position within the candidate chunks from Stage 1, followed by post-processing.

### Challenges and Learning Points:

As a newcomer to neural networks, my approach lacked certain elements:
- Random Dataloader: No ideas to implement a random chunk dataloader. Each epoch started from the same position, leading to overfitting and lack of robustness.
- Model Exploration: I used a very basic model (a single MLP, attention layer, LSTM, and head). Exploring more sophisticated models, as seen in tubo's work, would have been beneficial.

<hr>

## Concluding Thoughts:
This competition was a tremendous learning experience. My thanks to Kaggle for this opportunity, and to all the participants for their contributions and shared insights
