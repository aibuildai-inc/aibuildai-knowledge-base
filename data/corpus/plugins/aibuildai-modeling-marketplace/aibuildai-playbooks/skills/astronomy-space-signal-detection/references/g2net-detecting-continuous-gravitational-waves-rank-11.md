# 11th place solution OR Simpler is Better

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #11
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/375981

*First, I would like to thank the Armed Forces of Ukraine, the Security Service of Ukraine, the Defence Intelligence of Ukraine, State Emergency Service of Ukraine for providing safety and security to participate in this great competition, completing this work, and helping science, technology, and business not stop and move forward.*

Also thanks Kaggle team and competition organizers for this year G2Net Detecting Continuous Gravitational Waves event.

This year as in the previous one, we were working with simulated GW signals, but this year we had only STFTs of these signals, which makes this year's task a bit different

# Data 
## Data generation
After the first experiments with only training data, I realized that it is nearly impossible to build a good deep-learning model on it and validate solutions using Roc-Auc metric. So I have decided to start generating it using `pyfstat` library.
I have started from train set negative samples: I have used `*_timestamps_*` and `frequency_Hz` from the original train samples, as for other parameters, I have mostly taken them from one of the data generation Kernels:
```
generation_kwargs = {
        "sqrtSX": 0.5e-23,
        "detectors": "H1,L1",
        "Tsft": 1800,
        "Band": 1 / 5.01,
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.001,
    }
```
I have generated 100 duplicates for each training sample and applied adversarial validation with my best (CNN-based) model and received 0.5 Roc-Auc, which indicated that I was on a right way 
Then I repeated the same with positive train samples, using next additional parameters
```
random_param_dict = {
        "F1": lambda: 10 ** stats.uniform(-12, 4).rvs(),
        "cosi": lambda: stats.uniform(-1.0, 1.0).rvs(),
        "psi": lambda: stats.uniform(-0.25 * np.pi, 0.25 * np.pi).rvs(),
        "phi": lambda: stats.uniform(0, 2 * np.pi).rvs(),
        "Alpha": lambda: stats.uniform(0, 3.14159).rvs(),
        "Delta": lambda: stats.uniform(0, 3.14159).rvs(),
        "h0": lambda h0_center_: h0_center_ / stats.uniform(5, 90).rvs(),
    }
```
One more important step was to shift signal while generation, so it is not vertically centered. I have repeated adversarial validation and reached ~0.5 Roc-Auc again 
The next step was to repeat the same operation with the test set. But it was more tricky. The test set contains:
- Nonstationary noise
- Instrumental artifacts
 I have decided to treat only `Nonstationary noise` problem. I have found nearly all nonstationary samples with simple std computation and several iterations of adversarial validation (train vs test sets)
For stationary samples, I have repeated train set generation operation (10 positive and 10 negative duplicates)
For nonstationary I have simply used Gaussian distribution to generate test sample duplicate
```
def gen_multivar(ref_sample):
    locs = ref_sample.mean(axis=0)
    stds = ref_sample.std(axis=0)
    n_freqs = ref_sample.shape[0]
    return np.stack([np.random.normal(loc=loc, scale=scale, size=n_freqs) for loc, scale in zip(locs, stds)], axis=-1)


def gen_multivar_complex(ref_sample):
    return gen_multivar(ref_sample.real) + gen_multivar(ref_sample.imag) * 1j
```
Then I repeated adversarial validation with all generated data (~240K samples) and received ~0.5 Roc-Auc. So I was pretty much ready for modeling!
## Data preprocessing
I have tried 2 ways for data preprocessing:
1. Baseline with binning - described [here](https://www.kaggle.com/code/junkoda/basic-spectrogram-image-classification). 128 bins worked best for me
2. Laeyoung Data preprocessing with noise injection - described [here](https://www.kaggle.com/code/laeyoung/g2net-large-kernel-inference). Outperformed first option but only with huge amount of noise TTA on inference (64 noise TTA x 4 flips TTA)
# Modelling 
# Classification
I have mostly used simple CNN for spectrogram classification. As for backbones, I have tried several ones but the best was - `convnext_base_384_in22ft1k`.
I have trained model in the vanila image classification setup - Adam + ReduceLROnPlateau. But augmentations were pretty important:
- HorizontalFlip
- VerticalFlip
- ChannelShuffle
- ShiftScaleRotate (only by Y axis)
- OR Mixup 
After that, I have tried to finetune model without most of augmentations (reducing LR 10 times)
On inference I have just averaged 5 folds and used 4 flips TTA x 64 noise TTA. Inference in such a setup took several hours :( 
Such a model with  `Laeyoung Data preprocessing` granted me 0.764 Public and 0.785 Private scores (Not selected)
## Ensembling
For final `classification` ensemble I have blended nearly 10 models with different preprocessing types, architectures and augmentation setups
## Segmentation 
I have also trained a bunch of segmentation models using generated data. I have tried 2 setups:
- Use original SNR from generation 
- Normalize each spec with MinMaxScaler and apply some random SNR distribution for training and constant coef for validation 
Segmentation was not a silver bullet BUT it could catch some hard samples, which were ignored by classification!

So in order to use it in final ensemble - I have tried 2 approaches:
- Hard:
```
classification_probs[(masks > pixel_tresh).sum() > n_pixels] += prob_boost
```
I have used different `pixel_tresh` and `n_pixels` for different segmentation models but I have only used 1.0 as a `prob_boost` value. I have repeated such a `Probability Boosting` operation several times. Unfortunately, such an approach led to a huge overfit -  Scored 0.781 Public and 0.777 Private LB
- Soft:
```
segmentation_predict =  (masks > pixel_tresh).sum()
segmentation_predict[segmentation_predict < n_pixels] = -1
segmentation_predict = rankdata(segmentation_predict)
final_probs = segmentation_predict + rankdata(classification_probs)
``` 
And my final best selected sub contained soft approach with big classification ensemble and 5 segmentation models (blended with each other with `Soft` approach). Scored 0.776 Public and 0.781 Private LB


# P.S.
It was a long road to GM and I am very satisfied with this achievement. I would like to say  **Thanks** to all my teammates from previous competitions, all my tutors, Kaggle Community, and, of course, Ukraine, who has provided me with high-quality higher education, a competitive environment, and all needed opportunities
# Glory to Ukraine!!!
