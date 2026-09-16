# 4th Place Solution Brief Summary : Magic of 1D CNN

Competition: g2net-gravitational-wave-detection
Rank: #4
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275331

Hi all ,
First of all I want to thank the organisers and kaggle team for organising such a wonderful and interesting competition , we learned a lot . <b> I would like to thank <a>JarvisLabs.ai </a> (a GPU cloud based platform offering modern and extremely easy to launch GPU instances) for helping us during the competition by providing modern GPU cards. The platform enabled us to do multiple experiments rapidly with instant GPU instances. All our models were trained on <a href = "https://cloud.jarvislabs.ai/">cloud.jarvislabs.ai</a> GPU instances and this could not have been achieved without them. </b>

It was really a tough fight , we would have liked to have finished in the money but nevertheless we are really happy with our finish. It was lovely to once again team up with @nischaydnk @benihime91
@pheadrus and @proletheus

Similar to other teams we also started with constant Q Transforms and 2D CNN architectures , we did a lot of experiments with different preprocessing techniques (bandpass, whitening , denoising AE's , CWT ,etc details to be shared later) , this allowed us to reach top 15x .

Thanks to heng and other kaggler's experimentation we realized how well conv1d and sequence models are working on this data and it also intuitively made sense to us , so we started working on that and a custom 1d cnn architecture forms the main backbone of our solution

# 1D/Sequence Model Magic

We are really happy to announce that our single custom 1d architecture model scores <b> 0.8838 public LB / 0.8823 private LB and is in the gold zone alone </b>.  (Hoping to write a paper around it)

We started with a rather simple Conv1d architecture with just 8 conv1d layers along with a normal 2x linear head , to our surprise it scored really well cv 0.8766 Lb 0.8788 . This encouraged us to experiment more with sequence models , we tried a mix of LSTM's , GRU's , transformers ,etc but were not able to beat the normal conv1d model . 

Finally we decided to train deep conv1d model with residuals (similar to resnet) and it worked like a charm , we then changed the head from linear to LSTM and got further boost . Our final model architecture has the following flow :

GW waves numpy array --> horizontal stacking all three to get (1,4096*3) array --> band pass filtering ---> Deep conv1d backbone with residuals --->LSTM head ---> Prediction

We tried GRU , transformer and bert like heads but LSTM worked best

# 2D Model 

Most of our strong 2d models came from @proletheus who was at 15th position back then when we merged . He used a mix of augmentations and good normalization technique that gave us a good amount of boost in 2d models . We used both CQT and CWT based models in our final ensemble . 

We mainly used nnAudio CQT1992v2 and CQT2010 during the preprocessing with 
config.    

```
qtransform_params={"sr": 2048, "fmin": 30, "fmax": 400, "hop_length": 4, 
"bins_per_octave": 12, "filter_scale" : 0.3}
```

Sequence of Preprocessing is as follows:
Numpy ---> signal tukey ---> band pass filter ---> normalized by norm_by =[7.729773e-21,8.228142e-21, 8.750003e-21] --->CQT--> Augmentations[coloredNoise and shift]

Torch audiomentations were used to apply augmentations. Colored noise augmentation was done channel wise while shift was applied sample wise.


Please note that this is a small gist of our solution/journey .
@benihime91 will be publishing a detailed solution/explanation with code by tomorrow 

Thanks for reading
