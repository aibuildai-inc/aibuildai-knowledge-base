# 3rd place solution: How far can deep learning go?

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #3
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/376233

First of all, I would like to express deep gratitude to the competition organisers and the Kaggle team. The competition covers a wide range of knowledge, including data generation, deep learning, and classical methods, making it one of the most challenging and interesting competitions in my own Kaggle history. Without my wonderful teammates this ranking would not have been possible and it was indeed a pleasure for me to work with them.
Congratulations to all the winners (and a special big congratulations to the new GM and my amazing teammate Gleb)! It was very interesting to read about various methods used in other teams' solutions.

Our approach is comprised of an ensemble of various **deep learning** models and  **"creative denoising"**, which is a denoising technique that exploits the frequency overlap of real-world noise.

## Data generation
### Fixed dataset approach
We generated 40k images, including stationary noise, non-stationary noise, glitches (horizontal instrument lines) using pyfstat.

### Infinite training approach
As we look into the data, we found that background noise follows a chi square distribution (the real and imaginary parts follow independent gaussian distributions). We therefore considered a fast online generation of noise with almost identical statistics (mean, variance) to each test data ([This notebook](https://www.kaggle.com/code/analokamus/fast-online-noise-generation/notebook?scriptVersionId=115544433) shows how we generated noise). 
We generated 32k to 128k signal templates using pyfstat with sqrtSX=0, and injected signals into the generated noise. Also, since signals has frequency dependency (**Figure 1**), during training phase we: 
1. Sample a test image
2. Generate noise based on its statistics
3. Sample a signal generated at the same frequency from signal banks and inject

In this way, we were able to train our models with almost infinite background noise patterns. This prevented the model from overfitting to the background noise, and improved the model performance significantly.

**Figure 1.**


## Validation stratedy
We generated a validation set using the infinite training approach described above, with a one-to-one correspondence to the test set. Positive ratio was .66, and the injected signals has random parameters with signal depth ranged from 25 to 50. This validation set had good correlation with public LB. 

## Preprocessing 
### Temporal synchronization and resize
We synchronized the spectrogram from two detectors so that the model can catch the signal much easier, with no need to look around. Temporally averaged bins (128 - 720 bins) were used as the input image.

### Spectrogram normalization
#### Global normalization
( @chris62 )
We normalized the entire input image by subtracting mean and dividing by the std dev

#### Column-wise sqrt normalization
( @iafoss @analokamus )
Since background noise follows chi square distribution, we first calculated sqrt of the input image so that the output should follow normal distribution, and then subtracted mean / divided by std in each column to mitigate the impact of nonstationary noise.

## Architectures and training
### Spectrogram image classifier
We used CNNs with the following modification:
- Denoising layer(s) in the beginning with large kernels (3x31, 5x15, 5x31, 7x65)
- AE denoising based pretraining for large kernel layers ( @Iafoss )
- Denoising AUX with a light Segformer style decoder and MSE loss for noise-free signal ( @Iafoss )
- Add a UNet segmentation decoder to extract signal and use dice loss as aux loss ( @analokamus )
- Signal depth scheduling: start with 10-20 depth and finish with 25-50 depth ( @Iafoss @analokamus )
- Add frequency to the output head ( @chris62 )
- Incorporate frequency information and statistics based features ( @drhb )

Various backbones including EfficientNetB7, ResNeXt50, Inception v4, Xception65, Convnext base and ViT were used. 

| Model name | Public LB | Private LB | Author |
| --- | --- | --- | --- |
| VIT FREQ | **0.779** | **0.793** | @drhb |
```
- first 2 convolution layers from inception -> image_features
- [frequency, stat based features ] -> embedding 
- combining [image_features, embedding]
- applying this to `vit_large_patch16_224_in21k`
- trained for 30 epoch
- lr: 1e-4, cosine annealing
- augmentation: flip, time/freq masking
``` 

| Model name | Public LB | Private LB | Author |
| --- | --- | --- | --- |
| Model03 | 0.774 | 0.792 | @analokamus |
```
- infinite training with 80k signal templates
- input image size = 360 x 360
- 3x31 denoising layer
- UNet EfficientNet-b7
- loss = BCE loss x 0.7 + dice loss x 0.3
- cosine annealing lr for 40 epochs
- H/V flip, time/freq masking, random amplifier
``` 

### Matched filter-like classifier
( @analokmaus )
I randomly sampled 4096 or 8192 signals from the signal bank, and used them as convolution kernels. 
The output image with size of 360 x {num filter} is then passed to a simple CNN classification head.

| Model name | Public LB | Private LB | Author |
| --- | --- | --- | --- |
| Model03 | 0.768 | 0.782 | @analokamus |
```
- infinite training with 80k signal templates
- input image size = 360 x 360
- 4096 x 135 x 360 filters
- EfficientNet-b0 head
- BCE loss
- cosine annealing lr for 40 epochs
- H/V flip, time/freq masking, random amplifier
``` 

### Megaconv
( @bakeryproducts ) WIP

## Ensemble
### Hierarchical stacking
( @analokmaus @drhb)
To combine tons of predictions, we used the following approach: 
1. Separate predictions into stationary and nonstationary, this is because we observed different correlation patterns between predictions in noise types.
2. Categorize predictions into some groups based on the similarity of their training settings and fit a ridge regression model (large alpha regularization, positive coefficients only) to maximize AUC of the validation set inside each group (first stage stacking).
3. Collect results from first stage stacking (shaped {num sample} x {num group}), and fit another ridge regression (large alpha regularization, positive coefficients only) to maximize AUC of the validation set. 
4. Do 2-3 for both noise types defined in 1

The purpose of this two-stage stacking is to deal with multicollinearity and prevent the coefficients from falling to zero. We used two different ensemble profiles, which resulted in public/private score of 0.784/0.780 and 0.783/0.798 respectively.

### Bayesian-like approach
( @iafoss )
Consideration of 2048 random 4-fold splits, fitting 8192 linear models with a weak regularization, evaluation of weights of each model as an average over the final weight distribution.
This approch resulted in public/private score of 0.784/0.801.

Our final submission is the rank average of the three ensemble results described above, scoring **0.784 at public LB and 0.801 at private LB**. This is equivalent to **5th place** without the denoising technique (leak) we will mention below.

## Denoising (or leak 😉?)
### Internal denoising
( @iafoss @analokamus )
[This notebook](https://www.kaggle.com/code/iafoss/creative-denoising-part-of-top-3-solution/notebook) explains our denoising algorithm.

Four days before the end of the competition we realized that the nonstationary noise test data was generated by taking a large single chunk of real detector noise data, adding some white noise, a random selection of a subset of time_ids, sorting them, assigning randomly selected time_stamps, selection of the frequency range, and adding the signal. The frequency also appeared to be shifted by an integer number of Hz.

We went through all nonstationary noise test samples, performed a search between overlapping frequencies as all vs all bin match, found identical bins,  and computed abs1 - abs2, which gives a noise-free signal. 

One challenge after denoising is that we need to detect the origin of signal from the matched pairs. We summed spectrogram along time axis of the denoised image (abs1 - abs2), then calculated positive area and negative area. Big positive area suggest that the signal is from abs1 and negative area vice versa. Applying an adequate threshold made it possible to detect the origin of signal very accurately (**Figure 2**).

**Figure 2**. (left: denoised image, right: positive/negative area, top: no signal, middle: one signal, bottom: two signals)




We were able to denoise approximately 1100 samples, though in most cases the frequency coverage is partial. Therefore, the denoised data cannot give 100% confidence regarding the labels. We interpolated the prediction and denoised prediction based on the coverage. Overall this denoising improved our best ensemble from 0.784 to 0.807 at public LB, from 0.801 to **0.826** at private LB. 

### External denoising
( @bakeryproducts ) WIP

## Things did not work
- Noise to Noise denoising
- Noise to Void denoising
- Diffusion models
- Advanced stacking of the models with adding extra features
