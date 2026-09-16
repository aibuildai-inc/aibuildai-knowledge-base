# 9th place solution

Competition: asl-signs
Rank: #9
Source: https://www.kaggle.com/c/asl-signs/discussion/406343

Thanks Google and Kaggle for organizing this competition, I hope our solutions will help to create a PopSign product and lower the entering barrier to ASL. 

Below I'll give a brief overview of our solution, but before that I want to congratulate my teammate @timriggins, with this competition he became Competition Grandmaster! The fun part is that we met during Kaggle Days World Championship in Barcelona last October, once again thanks HP and Kaggle Days for such an opportunity :D

## Model

We used a transformer model which is very similar to the one @hengck23 shared
Things we changed in architecture/features:
* 1 -> 2 transformer layers (in the final ensemble we used 3 1-layer models and 1 2-layer model)
* aside from motion and distance features for hands, our models use lip and eye distances, 
absolute hand coordinates ```abdcoords_r = (rhand - np.nanmean(rhand, 0))``` 
and acceleration ``` racc = np.pad(rh_dist[1:] - rh_dist[:-1], [[0,1], [0,0], [0,0]]) ```

In two models we used absolute learnable (init. from sinusoidal) positional encoding, 
in other two we used learnable 1d-depthwise convolution as pe encoding 
```
self.pe = nn.Conv1d(emb_dim, emb_dim, kernel_size=5, padding=2, stride=1, groups=emb_dim)
## x.shape (B, L, E)
x = self.pe(x.permute(0, 2, 1)).permute(0, 2, 1) + x
```
## Augmentations

Empirically we found following augmentations to be effective:
* flip-aug
* random fps drop ```xyz = xyz[::2]```
* affine 3d - rotate around spine
* early mixup (keypoint space) & manifold mixup (embedding space)
* random cuts from start-end of the frame series
* interpolation

Mix-up was an important augmentation, it allowed us to use longer training schedule w/ cosine scheduler without overfit, all of our models were trained for 150 epochs and converged on the last epoch.

## Pytorch to TFLite conversion

We used [nobuco](https://github.com/AlexanderLutsenko/nobuco), check it out, it is very intuitive 

## Fun bug

I made a bug in a code that led to a better performance, when implementing a mixup augmentation I did this in the model forward pass function, - 
``` 
if np.random.randn() < 0.5 and mode_flag == 'train':
	x, labels_ohe = self.mixup(x,labels_ohe)
   ```
Later I found out that I'm using normal distribution instead of a uniform one thus if we imagine a PDF of normal disribution, we can say that we're using mixup in ~70% of cases, and when I corrected it, the performance dropped! (so it's better to use mixup with higher probability) 


## Results

Best single model reaches 0.7316 on the hengck participant split, 0.754 if using 4-model ensemble, CV/LB correlation was perfect and each CV improvement translated to both public and private LB. Single model gets scored in approx. 30 mins and ensemble gets scored very close to an hour. I was very worried, since we made last submission on the final day of the competition, but thankfully it didn't timeout :D

## What we tried, but didn't work
* External data, - WLASL-pretrain/adding WLASL to train/pretrain on greek sign language dataset, neither of those gave performance boost, even decreased compared to original initialization
* Continous MLM/deep predictive coding pretraining - I think that the model is too shallow to benefit for pre-training, I would like to hear from other participants if they used it in their solution
* Using 1d-CNN model, - we tried 1d-cnn with depthwise-separable convolutions 
* Arcface
* Train-time overparametrization (for linear layers), concept description can be found in MobileOne paper
