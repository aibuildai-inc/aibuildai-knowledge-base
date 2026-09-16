# 15th place: Two-stage model

Competition: waveform-inversion
Rank: #15
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587511

I thank the organizer and Kaggle for hosting this interesting physics competition.

My solution is a 2-stage model:

1. I use Bartley's caformer model, as the 1st stage, to predict the target sound speed field y.
2. Then I simulate signal x1 = Foward[y1] with the forward model using the 1st-stage prediction y1.
3. The input for the 2nd stage is 10-channel, concatenating the original signal x, and
the deviation of the simulated signal from the original: δx = x - x1.
4. The other input is the predicted sound speed field y1.
5. The two inputs are added up after stem convolutions and enter the caformer model in the Bartley model.
6. The target is the deviation from the input: δy = y - y1.



My imagination behind is that the sound speed field, y, is crucial for mapping time to depth, and the model must want to know it from the beginning.

The residual signal, δx, tells the discrepancy between the true sound speed and the estimated sound speed, and it should be informative how the model improve the previously estimated field y1.

I also had the diffusion model in mind, and I wanted to estimate the sound speed field iteratively: y_n -> y_n+1, but I could not make it work:

- I could not generate perturbed data y_n for training that generalize to the 1st-stage output
- So I use the out-of-fold predictions of the 5-fold 1st-stage model as the input for the 2nd stage
- I also tried the 3rd stage, but that did not improve the prediction. Maybe y1 is sufficiant and slightly better y2 does not help, but I am not convinced.

1. The 1st-stage model yields a validation score 27 (or ~30 for proper weight imitating the test set) and public score 27.9 for 5-fold median. 
2. The second-stage model reduces the error to Local CV 13 for a single prediction,  11 when I use 5 outputs from the 1st stage, and Public score of 13. I was convinced that this is a gold model solution, 6 am (Japan time) this morning, 9 am is the deadline, but to my surprise, there were 1-submission Grandmasters, not only one but three!!!



1st stage is trained for 120 epochs and 2nd stage is 125 epochs. The 2nd-stage output δy=0 corresponds to the 1st-stage solution, so the 2nd-stage model starts from the 1st-stage score by construction, although the model starts from new weights.

True flip augmentation:
I did not like the flip augmentation because the source at the center (channel 2) is at 34 and flip maps to 69 - 34 = 35, one pixel shifted. I simulated the signal for the flipped field with the same source position at 34 and randomly selected one of them during training.
