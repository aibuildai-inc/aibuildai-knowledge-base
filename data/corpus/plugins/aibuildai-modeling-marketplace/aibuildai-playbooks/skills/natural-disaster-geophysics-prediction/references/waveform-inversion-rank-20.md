# 20th solution - Caformer + data generation

Competition: waveform-inversion
Rank: #20
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587402

I want to thank Kaggle and the host for a great competition. I joined when @brendanartley shared his great notebooks,, esp the caformer one. Given I know nothing in computer vision I stuck to his model. The only variation i made was to also train it with the 5 input planes side by side, i.e. with a 1 channel, 1000x350 image. 

I won't describe my solution in detail as it is not a very good one, but here are few things I did that helped.

**Data augmentation.**

Besides the flip, I used a velocity scaling. If you multiply the velocity by alpha > 1, then you can shrink the seismic image by 1/alpha along the time dimension, and pad with 0. If you multiply velocity by alpha < 1, seismic data is expanded by 1/alpha, and we truncate it to the original size. This is basic physics. Well, I hope it is. I used this both at training time, and at test time. At training time, I randomly apply the transformation with alpha in [0.8, 1.2]. At test time, I computed the output with 10 values of alpha, and take the median of the outputs.

**Forward modeling.**

I found the code used by host to generate seismic data from velocity (https://github.com/lu-group/fourier-deeponet-fwi/blob/main/data/cfa/data_gen_loc_f.py), a week before it was shared by @hengck23 in the forum. This code can be used by batch, and I could run it on a single test file in one call. I always thought it would be great to leverage it. I tried various ways, including back propagation through it. It was too slow. What worked best was to generate additional data form test predictions. For each prediction, optionally apply some data augmentation to it (scaling + crop, velocity scaling, ect), then generate seismic data from it, then use the seismic data as input and the predicted velocity as target. There are a handful papers describing variants of this. The variants I used are the following ones. I started by generating 2 or 3 copies of test data that way, then tried an in batch method in the last few days. For that I add a batch from test data every 6 training batch. For that test batch I predict vel from the input. Then I generate seismic from vel, then predict on seismic to get a vel2 prediction. Then I back propagate the MAE between  vel2 and vel. This is quite effective. At the end I also ran this with only test data. This looked promising.

**Other backbones**

I let time fly and started looking at other backbones only a couple of days before end. Training them did not converge enough to help.

**Deep supervision**

I use the record (10 classes) as target with a linear head. The input to the linear head is the middle feature in unet, i.e. the one of size 9x9 in Bartley's model.  This helped quite a bit.

**What did not work**

A lot.

Really a lot. Main ones are these:

- To start with, my CV did not correlate anymore with LB when I reached a CV of 12 or so. I am not sure why, but it is why I submitted a lot. I used the same 20 files form the public notebook, maybe I should have used a larger and random set.

- I tried to use the dataset prediction to route the decoding through one of 10 decoder, a bit like a mixture of experts. CV improved quite a bit, but LB didn't. I should have revisited it in the end maybe, given my dataset prediction accuracy was 99.55% on the validation data.

- I generated lots of data using frequency and location moves as in https://github.com/lu-group/fourier-deeponet-fwi/blob/main/data/cfa/data_gen_loc_f.py but training with it took ages, and led to a LBimprovement of 1. I now see that I should have reused it at the end most probably, but I didn't for lack of time.

**My Takeaway**

Entering a computer vision competition solo was a bit silly given how little I know there. But this was a great learning experience, and I am sure I'll learn a lot from the amazing solutions from top teams.
