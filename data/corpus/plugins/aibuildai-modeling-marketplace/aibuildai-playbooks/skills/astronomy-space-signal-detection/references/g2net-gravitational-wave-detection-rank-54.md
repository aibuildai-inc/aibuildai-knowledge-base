# Public 69th / Private 54th solution

Competition: g2net-gravitational-wave-detection
Rank: #54
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275360

## Acknowledgements

Thanks to Kaggle and the hosts for holding this exciting competition! I've learned many things in this competition. Also thanks to all participants especially my teammate schulta( @schulta ) and Chizuchizu( @chizuchizu )

## Models

Scores of each model were as follows;

| Model | CV | LB |
| :---: | :---: | :---: |
| EfficientB7ns | 0.87706 | 0.8798 |
| EfficientB3ns | 0.87267 | 0.8747 |
| 1dCNN | 0.87281 | 0.8769 |
| swin transformer | 0.86977 | 0.8723 |

Detail of each model was as follows;

**CQT**
- EfficientNetB7ns
   - cpt parameter: {“sr”: 2048, “fmin”: 20, “fmax”: 500, “hop_length”: 8, “bins_per_octave”: 12, “filter_scale”: 0.7}
   - 5fold validation
   - Image size: 512 x 512
   - Tripret Attention ([paper](https://arxiv.org/abs/2010.03045))
   - Augmentation: shift in x-axis direction
- Swin Transformer
   - cqt parameter {“sr”: 2048, “fmin”: 20, “fmax”: 500, “hop_length”: 8, “bins_per_octave”: 12, “filter_scale”: 0.7}
   - 5fold validation
   - Image size: 384 x 384
   - Augmentation: shift in x-axis direction

**CWT**
- EfficientNetB3ns
- 5fold validation
- image size : 256 x 256
- Tripret Attention 
- Augmentation: shift in x-axis direction

**1dCNN**
- We used [Public Kernel](https://www.kaggle.com/scaomath/g2net-1d-cnn-gem-pool-pytorch-train-inference) as a baseline.Thanks to Shuhao Cao( @scaomath )
- 5fold validation
- add one more conv layer
- bandpass_params = dict(lf=25, hf=500)

## Other things
**Worked**
- SAM optimizer
- Normalization (separately for each gravitational wave observatory)
- align spectrograms from each gravitational wave interferometers on channel axis

**Not Worked**
- Mix up
- VQT
- pretrain using SETI data

**Idea**
- matched filtering
- 2 stage learning
- denoising auto-encoder
  
If you have a question about this solution, feel free to ask!
Thank you for reading!
