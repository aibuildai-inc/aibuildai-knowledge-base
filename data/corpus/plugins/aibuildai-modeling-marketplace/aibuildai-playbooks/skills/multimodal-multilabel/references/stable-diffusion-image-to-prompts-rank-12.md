# 12th place solution - GPUs go Brrr

Competition: stable-diffusion-image-to-prompts
Rank: #12
Source: https://www.kaggle.com/c/stable-diffusion-image-to-prompts/discussion/410657



YAY this is my first solo gold medal! Thanks Kaggle for organizing this! As always it was a fun but demanding competition. Also huge thanks to my employer, [freepik.com](http://freepik.com) for the compute. I would have never made it with my own equipment!

My strategy all along was to amass as many images and train as many models as possible in the 3 months span of the competition.

# Data

I generated roughly 4.4M images. Initially I used the script referred to by the organizer, but a few weeks into the competition I adopted a few optimizations from pytorch 2.0 and the diffusers library, such as memory efficient attention and `torch.compile` and rolled out my own generator largely based on Pedro Cuenca's [accelerated diffusers blog post](https://pytorch.org/blog/accelerated-diffusers-pt-20/). I used the same generation procedure as the organizer, or at least the bits that were public: generate 768px images then rescale to 512px, same CFG, noise sampler, etc. It would have been much faster to get 512px images straight out of the model but I hypothesized that the former would approximate the test set domain better. I kept the images in lossless format (WebP). Using JPGs was much more manageable but resulted in slightly worse LB.

I tried to gather prompts as diverse as possible including real image captions from LAION, CoCo, etc. and scraped prompts (DB2, OP, MP):

- Conceptual Captions (~2M)
- Open Prompts (~1.2M)
- CoCo (~0.4M)
- Magic Prompt (~0.4M)
- DiffusionDB2 (~0.4M)
- LAION aesthetics (~0.1M)

I used @xiaozhouwang 's idea to [filter out images with cosine similarities >.9](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/398529) as many prompts, especially those scraped from discord, etc. are simply a rephrasing of another prompt. I also discarded prompts longer than 77 tokens. Deduplication had a significant impact in both CV and LB.

2023-08-14 EDIT: [Link to the full train/validation dataset](https://huggingface.co/datasets/jamarju/sd-4.4M).

# Single models

I tried many architectures with a 100K subset distributed more or less uniformly from the Open Prompts database, and then selected the following models for the final train:

- eva_giant_patch14_336 (1B params)
- eva02_large_patch14_448 (300M params)
- eva02_large_patch14_448.mim_m38m_ft_in22k (300M params)
- vit_huge_patch14_clip_224.laion2b  (630M params)
- vit_large_patch14_clip_336.openai_ft_in12k_in1k (300M params)

I used timm's pretrained weights with a single `Linear(x, 384)` head and negative cosine similarity loss. Some of those models were released in the middle of the competition, so enormous thanks go to Ross Wightman for his titanic and continuous job.

I trained for 3 epochs with OneCycleLR schedule and lr=1e-6 for most of the models and Adam, 8-bit Adam from bitsandbytes or 8-bit Lion (also from bitsandbytes)  depending on the model and deepspeed stage 2 for distributed training. Overall 8-bit training worked fine but some models were only competitive when trained with regular Adam + AMP (bfloat16). Each architecture was trained with a different random seed. In the last couple of weeks I retrained 10 of the weaker models using constant LR + warm up as some models improved both CV and LR over their OneCycleLR counterparts, but that was not always true so I kept the OneCycleLR checkpoint where it was better.

The total compute I used to train all 25 models was ~215 gpu\*days plus another ~85 gpu\*days taken by failed/suboptimal/interrupted trains. GPUs were mostly RTX 3090.

# Validation strategy

To tame the CV/LB correlation, I clustered the embeddings into 100 clusters using KMeans and ran K-fold validation with 20% holdout samples grouped by cluster. My CVs were consistently between 9% and 17% higher than the LB for single fold models, 13% on average. 

# Ensemble

For the final ensemble I trained a very simple `nn.Linear(5, 1, bias=False)` to find the optimum blending coefficients. I tried other sophisticated sample-wise strategies, such as a transformer encoder with or without architecture hinting or an MLP, but blending linearly worked better.

In the very last hours I had 1 spare sub, so I submitted just the EVAs dropping the two (weaker) ViTs and that turned out to be my best scoring submission.

# Pain points

Keeping track of so many models was the hardest part for me in this competition, as I am not an overly organized person. Also, some models training lasted for as long as 9 days! Murphy's law held true most of the time and I experienced many kinds of interruptions:

- power outages, 
- NVMe errors, 
- disk full at checkpoint save (ouch),
- Linux OOM killing sprees, 
- GPU thermal throttling, 
- higher prio GPU tasks (ie. other people wanting to use the computer for real work),
- even tmux has blown up in my face twice

If I had to chose my least favorite crash that would probably be tmux's.

I can't imagine what months long training of massive LLMs must be!

I used pytorch lightning + checkpoint callbacks every epoch AND every hour. Unfortunately, the hourly checkpoints (or any mid-epoch checkpoint for this matter) are not properly resumable because the dataloader is not deterministically fast-forwarded to the exact same step. That results in some images being run twice and some others not run at all, and therefore a lower CV.

# Things that didn't work

- LORA (trains fast but less competitive)
- Many other archs (ConvNext XL, XXL, efficientnet v2, smaller ViTs, BEiT, BEiTv2, vanilla resnets, etc).
- Patch dropout (trains really fast but less competitive)
- Tensor-rt (what a nightmare)
- Contrastive loss instead of -cosine similarity loss.
- GPT-3.5/-4 prompt generation (problems of obedience to the prompt and repetition of themes, but I didn't try too hard)

# Thanks

Again, Ross Wightman for timm, @ptrblck at the pytorch forums, my employer [freepik.com](http://freepik.com) (currently hiring, btw!) and the organizers.
