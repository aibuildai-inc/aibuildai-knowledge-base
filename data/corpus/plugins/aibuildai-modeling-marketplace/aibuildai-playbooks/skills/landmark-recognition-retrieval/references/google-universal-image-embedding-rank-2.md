# [2nd place] Solution

Competition: google-universal-image-embedding
Rank: #2
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359525

Firstly, many thanks to @andrefaraujo for hosting such an interesting competition. And I'm glad that I moved from 3rd to 2nd place final in this competition.( sixsixsix scored higher than me actually!). I will introduce my approach in shot.

[Inference Notebook](https://www.kaggle.com/code/w3579628328/2nd-place-solution?scriptVersionId=107642661)

#### Dataset
* Aliproducts
* Art_MET
* DeepFashion(Consumer-to-shop)
* DeepFashion2(hard-triplets)
* Fashion200K
* ICCV 2021 LargeFineFoodAI
* Food Recognition 2022
* JD_Products_10K
* Landmark2021
* Grocery Store
* rp2k
* Shopee
* Stanford_Cars
* Stanford_Products

I've used a large amount of data, but didn't do much selection.

#### Model
* open_clip_torch: Vit-H14-224-visual as backbone
* fc with dropout rate=0.2 as neck
* arcface head.
* Freeze backbone to finetune get me 0.662+-, flip boost it to 0.666.

#### Idea
* Data Sampling: Set maxnum_images_perclass=100, minnum_images_perclass=3.
* Data Resampling: Do lightly resampling to balance class weight, set a minnum_images_perclass=20.
* Dynamic Margin (0.666->0.670): From the 1st solution of shopee competition. Set init m=0.1, then m= m+0.1 *(cur_ep-1), max_m = 0.8.
* stratified lr(0.670->0.713): This is a huge leap in my experiments. To remain the representation ability of the pretrained  model, I set the init_lr to 1e-4, which is the lr for the none-backbone parts, and the lr for the backbone is **init_lr*lr_factor**(lr_factor is reasonably low), the experiments was still on the half way when competition ended. Anyway, **lr_factor=0.001** is kind of a nice one, which gives me a score of 0.713 finally.(finetune with lr_factor=0.001 for 2epochs, then freeze backbone to finetune for another 1 epoch, do flipping finally.)

#### Conclusion
To do experiments more efficiently, I just sampled about 1M data for training in the early stage. during the journey to 0.670, I’ve tried many different architectures, more complex neck, sub-center arcface head(works well in open/clip_L_336), backbone ensemble, etc., but the scores just moved slowly. So I thought 0.670 is kind of a bottleneck for me, freezing backbone, keeping my eyes on the none-backbone parts. Then I decided to turn to the backbone after having done every experiments I thought I should do, even I once failed(I tried unfreezing backbone and finetune simply, the score just drop down rapidly). After thinking kind of deeply of how the backbone works, and also with luck, I succeeded in my first experiment, got a score of 0.688. I've just tried 3 different lr_factor due to time limitation, it took me about 12h+ to train one epoch. Performance: 0.001(0.707)>0.005(0.700)>0.01.

Here I didn't do much explanation of why I did so, and there are some other interesting experiments I've done, maybe will even get a higher score. I'll include all these in paper, you are welcome to check out the fully solution if you are interested!🤗

It's my first time to step into finegrained Image world, which is really interesting! I've learned a lot from this competition, and also have some further understanding on this kind of task.
Finally, thanks to everyone that contribute to this competition!

Code:
https://github.com/XL-H/GUIE-2nd-Place-Solution

Paper:
https://arxiv.org/abs/2210.08735
