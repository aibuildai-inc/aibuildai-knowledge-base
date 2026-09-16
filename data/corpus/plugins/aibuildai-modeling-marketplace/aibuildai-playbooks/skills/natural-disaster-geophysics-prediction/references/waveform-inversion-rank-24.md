# 24th Place Silver Medal Solution

Competition: waveform-inversion
Rank: #24
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587390

Hi Everyone, I would like to share my solution as well as the trick I used to boost my score.

# Overview
- Model Architecture: [Unet with CAFormer Encoder](https://www.kaggle.com/code/brendanartley/caformer-full-resolution-improved)
- Public LB Score: 16.4
- Private LB Score: 16.4

# Introduction
Something that stood out for me during the competition was that an almost exact reproduction of the forward modelling simulation code used by the hosts was open-sourced [here](https://www.kaggle.com/code/manatoyo/improved-vel-to-seis) .

This means that given a set of velocity maps, we can use this code to accurately generate its corresponding seismic data pair. This allows us to create a synthetic dataset of extra data to train our model on.

Then comes the next question: What set of velocity maps should we use? One approach is to augment the existing velocity maps. Alternatively, we could train a generator model to create new velocity maps.

I pondered over this question for a long time, I wanted a set of velocity maps that had a potential to teach the model useful information that would greatly boost its performance.

Hmmm......If only I had access to the velocity maps of the test set. That would be great. But wait, I do have access to something similar! ***The model's predicted test set velocity maps.***

# The Trick: Pseudo-Labelling
Using the forward modelling simulation code, the model's predicted test set velocity maps can be used to reconstruct a synthetic test set that largely resembles the actual test set. Training the model on this synthetic test set greatly enhances performance on the actual test set.

Methodology:
1. Use current best model to predict test set velocity maps
	- Best model is selected based on feedback from LB
2. Use predicted test set velocity maps to reconstruct corresponding seismic pairs to create a synthetic test set
	- Running the forward modelling simulation code through all the test set velocity map predictions on one CPU instance takes extremely long. So I broke them up into subsets and ran them on multiple CPU instances
3. Train new model using synthetic test set
	- Initialise training with the current best model to reduce training time

This process can be repeated multiple times to further enhance performance. I will refer to each repetition as one "cycle".

Every time we get a better model, the predicted test set velocity maps also get closer to the actual test set velocity maps. Hence, the more cycles we run, the more similar our synthetic test set will be to the actual test set. This results in further performance increases every cycle.

The intuition behind this idea can be summarised as follows:
- Cycle 0 Model
	- Here are my predictions for the test set velocity maps.
- Synthetic test set
	- Unfortunately, your predictions are slightly off. Here is what the seismic data for the predicted test set velocity maps you have given me should look like.
- Cycle 1 Model
	- I see! I know what went wrong now! 
	- I have learnt from my mistakes. Here is my new predictions for the test set velocity maps.

Essentially, the forward modelling simulation code is acting as a teacher to the model. This is only possible due how well the forward modelling simulation code can reconstruct the seismic data from the velocity maps.

I ran a total of 6 cycles (cycle 1 to cycle 6) and the results are shown below.


# Other notes
I made certain changes to the methodology detailed above in certain cycles mainly because I wanted to try out some ideas that I thought were promising:
- Cycle1: Additionally included the entire original OpenFWI dataset
- Cycle4: Additionally included data from the previous 3 cycles as well
- Cycle5: Additionally included approximately 14% of the original OpenFWI dataset
- Cycle6: Additionally included Style_A, Style_B, and CurveFault_B data from the original OpenFWI dataset

However, in hindsight, I believe that these changes were unnecessary and training the model on only the synthetic test set for all cycles would probably have yielded the same results.

Key training parameters adjustments:
- Epochs --> Adjusted to take into account both the total size of the training data and my sleep schedule. Ensuring that the code runs in reasonable time and that I am always running code 24/7.
- Learning rate: 1e-4 --> Lowered to keep training stable and prevent NaNs
- ConstantCosineLR pct_cosine: 0.9 --> Increased to start decreasing learning rate earlier to account for the fact that we are using a pre-trained model instead of training from scratch.

# Closing Remarks
This competition has been quite a journey, full of ups and downs. 

Early on, I had several ideas I was eager to explore, but unfortunately, most of them didn’t work out as I’d hoped. Despite the setbacks, I kept experimenting until I finally discovered the approach that significantly improved my score. 

For a while, I thought I might have a real shot at earning my first gold medal. However, in the final stretch, many teams surged up the leaderboard, leaving me behind.

Even so, this has been a valuable learning experience, and I’m determined to come back stronger next time.

I’m looking forward to seeing what strategies the top teams used.

# Credits
[CAFormer - Full Resolution Improved](https://www.kaggle.com/code/brendanartley/caformer-full-resolution-improved) by @brendanartley
[waveform-inversion : Vel to Seis](https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis) by @jaewook704
[Improved Vel to Seis](https://www.kaggle.com/code/manatoyo/improved-vel-to-seis) by @manatoyo

# Code and Other Material
[Inference Notebook](https://www.kaggle.com/code/sshiyu/final-submission-caformer-inference-only)
[Models and Training Codes](https://www.kaggle.com/datasets/sshiyu/gwi-models-and-training-codes)
[Best Synthetic Test Set](https://www.kaggle.com/datasets/sshiyu/gwi-synthetic-test-set)
