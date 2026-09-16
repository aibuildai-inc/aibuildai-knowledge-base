# 5th place solution (instead of)

Competition: g2net-gravitational-wave-detection
Rank: #5
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275334

Thanks to kaggle, organizers, and community for intense battle at the end and live discussion in the beginning.

Not really a solution post, rather rant on EfficientNets. Solution itself is pretty straightforward, just 2d CNN go brrrrrr.

# TLDR

- PSD whitening with precomputed mean PSD for each channel, custom window function : mirrored sigmoid
- 1d augs: shift, noise, mixup, cutmix, shuffle
- 2d CQT default settings fmin=16, fmax=1024, hop=12
- 5xCNN 2d ensemble with TTA, models and EMA's
- EfficientNets are not so good (for me).

# Intro

As i don't know anything about competition problem and do not have any experience at 1d-audio-spectrogram-ish type of tasks: I decided to go full research mode on specific NN architecture

Disclaimer: CNN shouldn't be the right fit for spectrograms, but they are. I also tried Transformers, surely they can work, but score-wise i was barely able to 
go into 88.+ on LB. I'm still pretty new with transformers thou.

# EfficientNets

What a original choice, you might say. Indeed, a lot of people uses EN for some reason, but not me. I don't use EN family because it is not ... efficient.
It is not much of a secret and there are some papers that are pointing at that (i.e. RegNet, GENet, EffnetV2, etc). But what if I was wrong all that time and 
img/sec -to- acc ratio is much better? 

It is not.

My best result with EN after a lot of tuning and fitting was easily beaten (not by much) on the first try with RegNet and even ResNet. **And they are 4x times faster.**
Of course on some datasets EN will work better, but im still to find one, other then ImageNet1k.
There can be a lot of reasons why it is what is is, my thoughts on that: it is happening because EN is "overfitting" ImageNet1k (can be NAS or compound-scaling)

My best EN ensemble score was 8830 LB, after that I gave up and get ResNets into the mix in last couple of days of the competition.

Thanks for reading and please consider some other arch's if you dont already.
