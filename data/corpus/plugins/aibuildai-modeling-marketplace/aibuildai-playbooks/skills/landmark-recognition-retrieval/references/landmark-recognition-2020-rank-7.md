# 7th place solution (with inference kernel + training code)

Competition: landmark-recognition-2020
Rank: #7
Source: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187894

Firstly, I want to thank my teammates @aerdem4 and @dattran2346 for their hard work. In this comp., they focused on improving global models so that I could spend time on local part.

Our best sub. was geometric mean of similarities from 4 models (2 backbones SEResNext101+ResNext101-32x4d at 512x512 and 736x736 image sizes) + re-ranking by SuperPoint + SuperGlue. We pre-computed 1.6m train embeddings for 4 models, added them as ext. data and simply filtered for the relevant private image ids (100k) for inference.

**1. Global models**
We continued training from our best checkpoints from retrieval challenge (https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176151). Our models were originally trained with focal smoothing loss with a tweak to work nicely with CosFace (thanks to Ahmet) on GLv2 clean dataset. For this comp., we used all the remaining train images from 81k classes (3.2m images) for re-training. We also came across the seesaw loss paper (https://arxiv.org/abs/2008.10032) and decided to give it a try (thanks to Dat).
Model architecture was just simply CNN + GEM + Linear + BN + CosFace, like many other teams. 
Our 2 models were then re-trained with these two losses as following:
+ Stage 1: 20 epochs at size 512x512
+ Stage 2: froze batchnorm layers, then fine-tuned at size 736x736 for 2 epochs.
Compared to the retrieval comp., our local validation scores were boosted by 5% and correlated well with the LB. From our estimates, 3% was due to adding extra data, 1% from the seesaw loss and 1% from larger image size.

**2. Local model**
After obtaining the top-k nearest train ids for each test image, we used SuperPoint + HRNetv2 pre-trained on ADE20k dataset to filter predicted keypoints on sky/person/flowers/tree classes + SuperGlue (copied from the winning sol. at CVPR this year) to calculate the number of inliers between each image pair. Local score was then obtained using the formula provided by the host team (with max_num_inliers=200) and then multiplied with global score to get final score. This post-processing step gave a very strong boost (4-6% in public LB to our above global models). However, we noticed that the better the global models were, the less improvement this local re-ranking step gave us. In addition, we couldn't come up with a reliable strategy to evaluate SuperPoint+SuperGlue when integrating with global models locally. Relying solely on LB was pretty dangerous and we were lucky to stay at 7th 😅.
I also tried to re-train SuperPoint and SuperGlue on a subset of 200k clean train images but couldn't obtain any positive improvement.

Our kernel + training code + model checkpoints were here: https://www.kaggle.com/andy2709/fork-of-recognition-notebook-16367c-511fa4-95d3bf?scriptVersionId=43685736. 

P/s: The DELG weights + training config. files for 3 different backbones (res50, res101 and seres101) will be uploaded soon 😃. They were all trained with 512x512  images for 10 epochs with AdamW and cosine scheduler. In retrieval challenge, res50's perf was 0.299/ 0.268 (public/ private); others were higher so I'm quite confident in the implementation correctness.
