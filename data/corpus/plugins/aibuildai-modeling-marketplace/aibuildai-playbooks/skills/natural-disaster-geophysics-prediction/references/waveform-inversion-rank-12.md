# 12th Solution

Competition: waveform-inversion
Rank: #12
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587404

First, I’d like to thank the competition organizers, as well as Bartley. I started paying attention to this competition 18 days ago, and without Bartley’s baseline, I probably wouldn’t have participated.
Also, big thanks to my teammates for their efforts.

### Baseline

    Together with @carlosgdcj and @sjtuwangshuo, we trained a Stable Diffusion model to generate vel-to-seis data, producing about five times the amount of data compared to OpenFWI.
Based on Bartley’s model architecture, we applied min_max(log(x)) normalization on the inputs and trained a baseline model that achieved 19.3 on the leaderboard.

### Online Augmented Training

    After discovering the original vel-to-seis forward function, I re-implemented an accelerated version that could reach about 120 samples/s on a 5090 GPU  (opened here:  https://github.com/lhwcv/vel_forward_v2)
This enabled us to perform online data augmentation during training.
We mixed the training and test data, where the velocity labels of the test set were generated using the aforementioned function.
We also trained a simple classifier, which allowed us to split up the test set family by family so that each teammate could handle a few families.
At this stage, we improved our score to 13.2.

### Further Improvement on Test Set Only

    We then fine-tuned on mini-batches of test-only data using the previous model, where the validation was directly based on seis reconstruction error.
    This step brought our score down to 12.5.
    Finally, we directly optimized the velocity using the gradient-enabled FWM function on StyleA/B, without any model — just taking the model output as initialization and optimizing it directly.
This pushed our final score to 12.3.

### Summary

We truly enjoyed reading the Rank 1 solution.
Big congratulations to everyone for such interesting approaches and for keeping the leaderboard nearly shake-up free in the end.
