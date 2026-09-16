# 55th place solution - A frugal approach

Competition: bengaliai-cv19
Rank: #55
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136799

Thanks Kaggle and Bengali.ai for hosting this contest. Congratulations to all the winners and all the participants who kept the discussion alive and encouraging throughout the contest with few special shoutouts to @haqishen, @pestipeti, @hengck23, @ildoonet, @bibek777, @machinelp for really interesting discussions inundating with ideas. Special thanks to @iafoss for his starter notebook and @drhabib for posting detailed code of previous competition solutions, which I  relied upon. 

This is my first competition medal on Kaggle in 2 years. The public leaderboard standing is from one single model. Due to lack of hardware resources, most of my code was trained on Kaggle Kernels (The reason why I call this a frugal approach ;p). 

## Preprocessing and Augmentations
I resized the images to **240x240x1** and zero-padded them. Simple augmentations such *as crop_resize, image_wrapping, rotations, and image_lightening* were used. 

## Validation
I experimented with **5 fold CV** and for submission notebook, I performed a **80-20** split of the dataset. 

## Backbone and architecture
I performed most of the ideas shared on discussion forums keeping EfficientNet-B1 as the model backbone, which I took from EfficientNet Pytorch, and carried out certain modifications inspired from @iafoss's starter kernel. EfficientNet-B3 and above, being computationally expensive for Kaggle kernel, could not be trained properly in the limited kernel runtime of 32000s.

## What didn't work
Although MixUp and Cutmix augmentation seemed really promising, they provided no effective improvement for me. They require higher epochs for training which was not at all feasible in my case.

Based on some previous contest discussion advice, it was helpful to maintain experiment log files. I naively created the experiment log, but, it turned out to be confusing rather than helping when selecting the final solution for submission. 

Lack of hardware was a major turndown for me at the beginning of the contest and no improvement in the public lb was discouraging. The time that was wasted without any submission could have been used crucially. I timed every single epoch and then divided 32000s from that in order to train my model for longer and longer time. I used half precision and had to replace _Mish activation_ in the code with _ReLU_ for faster computation (It gave me almost one extra epoch). 

### What I learned during the competition ( note to self )
* Prepare and maintain a proper experiment log file throughout the contest (Any help with this would be appreciated)
* Discussion forums are really interesting and full of ideas to experiment with but proper experiment documentation is also necessary.
* Proper selection of solution for submission is extremely important, this helped me through LB shakeup.
