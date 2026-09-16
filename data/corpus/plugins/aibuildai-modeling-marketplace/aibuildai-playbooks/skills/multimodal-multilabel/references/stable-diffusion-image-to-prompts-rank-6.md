# 6th Place Solution

Competition: stable-diffusion-image-to-prompts
Rank: #6
Source: https://www.kaggle.com/c/stable-diffusion-image-to-prompts/discussion/410768

Thank you organizers for hosting this competition, and congrats to all the winners. I was depressed today after missing the prize, but I am happy to have won the gold medal.
I tried some approaches at the beginning of the competition, but after all the solution settled down to quite a simple one:
**create 8.87M SD2-generated images from the captions of the image/video captioning dataset, and after deduplicating/preprocessing the dataset, train the generated images with timm's latest backbones (i.e. eva, convnext, swin).**
The inference notebook for final submission is [here](https://www.kaggle.com/code/sokazaki/sdip-6th-place-submission)

**First Attempts**
At the beginning of this competition, as the public notebooks do, I also tried some image captioning models.
After trying several models, the combination of 4 captioning models ([GIT trained on MS COCO](https://arxiv.org/abs/2205.14100) / CoCa / [Socratic model from IC3](https://arxiv.org/abs/2302.01328) / CLIP Interrogator) and ViT trained model from [@shoheiazuma notebook](https://www.kaggle.com/code/shoheiazuma/stable-diffusion-vit-baseline-inference) reached 0.5568 Public LB, and with the 4 image captioning models + ViT&Swin trained using [@shoheiazuma notebook](https://www.kaggle.com/code/shoheiazuma/stable-diffusion-vit-baseline-train) the score reached 0.57439 Public LB.
Therefore, at the beginning I considered creating two ways (i.e. image captioning models and direct optimization models like [@shoheiazuma notebook](https://www.kaggle.com/code/shoheiazuma/stable-diffusion-vit-baseline-train)) to create final submissions, however, the inference time of image captioning models were too long, so I discarded to search for the best image captioning models, and shift to concentrate on creating direct optimization models. For creating direct optimization models, it was the key to create the SD2 dataset as soon as possible, so I started the survey of the image/video captioning dataset.

**Data Generation**
I surveyed the image captioning / video captioning dataset and selected the below datasets to create SD2 images.
![SD2 generated dataset] (https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1476759%2F8a506f048c85efde0c8b8894827b5cb5%2F2023-05-16%20231638.png?generation=1684246639225798&alt=media)
For the training dataset, The rich vocabulary & the moderate length of the text captioning seemed important to me, so I selected the dataset based on the two factors. Also, I used the faiss script from [@tomokihirose notebook](https://www.kaggle.com/code/tomokihirose/use-vector-search-for-diffusion-db-cleansing) only for Diffusion14M DB and WebVid10M to remove redundant above 0.9 cosine similarity captions from the dataset. Also, as preprocessing for the dataset, the captions which are not English / shorter than 5 words / longer than 70 words are removed from the training dataset.   
For empowering depth effect [as discussed in this thread] (https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/398407), I also added diffusion2M DB 224x224 resized dataset [15] and another created 722K (180.5K captions * 4 different seed) from diffusion2M DB captions to the above training dataset.
The gain by the depth was nice (+ 0.005～0.010 Public LB) as shown in the latter results, so I should have dug more deeper into the depth side for further performance improvement.

To create the SD2 dataset, I used the diffuser's library (the weight is from stabilityai/stable-diffusion-2) with the default setting. For a fast generation, I also applied torch.compile mode and xformers with PyTorch2.0 using the below snippets:
```python
stable_model.unet = torch.compile(stable_model.unet)
stable_model.enable_xformers_memory_efficient_attention()
```
For further speedup, applying [Token Merging] (https://github.com/facebookresearch/ToMe) or using DPMSolver++ in SD2 were candidates, however, I couldn't have confidence that using those approaches won't decrease the total performance and it's hard to recreate the dataset due to the time resources, so I decided not to use those techniques.
For creating 8.87M dataset, it took about 2 months with 5～10 V100s and 1～5 A100s. In my setting, V100/A100 can produce about 100K/200K images in one week, so it costs about 600 V100 days / 300 A100 days for creating 8.87M dataset.

**Model Training**
I selected 3 backbones ([eva-large 336 size](https://huggingface.co/timm/eva_large_patch14_336.in22k_ft_in22k_in1k), [convnext-large 384 size](https://huggingface.co/timm/convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384), [swin-large 384 size](https://huggingface.co/timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k)), and each model is trained for 5 epoch. For training, it took about 3～7 days on one A100. All models converged around 4 epochs for the validation dataset (train/valid split is 0.9/0.1).
In my experiments, the performance of eva 196 size, deit3, and efficientnet variants was lower than the above 3 models, so I didn't use them.

The training script is almost the same as [@shoheiazuma notebook](https://www.kaggle.com/code/shoheiazuma/stable-diffusion-vit-baseline-train). The difference from the training notebook is here:
- changed batch size to 128 for all models
- training with fp16 mode
- applied horizontal flip (ColorJitter, GaussianBlur, and RandomCrop didn't lead to the improvement)

Also, in the retraining phase, I changed CosineEmbeddingLoss() to CosineEmbeddingLoss(margin=-0.5) + 0.01 x MSELoss and trained with all datasets.
For final submission, the highlighted 14 models in the attached picture were ensembled with different resize resolution TTA (460x460 and 512x512) with equal ensemble weight.
The performance of each trained model is here. Split1 (5.02M) and Split2 (6.50M) are selected from the created 8.87M SD2 dataset except for 722K (180.5K * 4 different seeds) from DiffusionDB2M:

![ConvNext Results] (https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1476759%2F59e4ce02426f42578cef6d1cdca8bb5e%2F2023-05-18%20004915.png?generation=1684338631593500&alt=media)
![Swin Results] (https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1476759%2F628b5ae887ae41492ad40f6c4800c03e%2F2023-05-18%20004936.png?generation=1684338658637625&alt=media)
![EVA Results] (https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1476759%2Fd02d95a4f44090f657b420bb7ee45762%2F2023-05-18%20004955.png?generation=1684338689799294&alt=media)

**What didn't work in my environment**
- Hard Prompts Made Easy (https://arxiv.org/abs/2302.03668): 
the score using this method was low, and it took a bit long time to generate each optimized prompt.
- Stable unCLIP (https://github.com/Stability-AI/stablediffusion/blob/main/doc/UNCLIP.MD): 
I tried to use unCLIP for Test Time Augmentation, but it took a long time to generate each image variation.
- Training with Stable Diffusion's latent feature: 
Extracted the feature of VAE in SD2 to train, but it didn't work well.
- Using LLM (e.g. T5, GPT2, GPT3 Davinci-003) to summarize/regenerate the captions from image captioning models: 
I couldn't get much gain even with GPT3 Davinci-003, so I stopped to use this approach.
- Also I tried to generate prompts using Vicuna-13B, however, I couldn't get the weight from the application form, so I gave up using it.

**Acknowledgement**
I sincerely would like to appreciate Hitachi Ltd., as I used my company's resources for creating datasets and training models.

**Reference for the dataset**
[[1] MS COCO captions 2017](https://cocodataset.org/#download)
[[2] DiffusionDB 14M] (https://huggingface.co/datasets/poloclub/diffusiondb/tree/main)
[[3] Flickr Funny & Romantic dataset] (https://paperswithcode.com/dataset/flickrstyle10k)
[[4] Flickr 30K] (https://paperswithcode.com/dataset/flickr30k)
[[5] Iaprtc12] (https://www.imageclef.org/photodata)
[[6] Nocaps] (https://nocaps.org/download)
[[7] SBU Captioning Dataset] (https://www.cs.rice.edu/~vo9/sbucaptions/)
[[8] Senticap Dataset] (http://users.cecs.anu.edu.au/~u4534172/senticap.html)
[[9] Textcaps Dataset] (https://textvqa.org/textcaps/)
[[10] VizWiz Dataset] (https://vizwiz.org/tasks-and-datasets/image-captioning/)
[[11] VTT 2016-2022 from TRECVID] (https://trecvid.nist.gov/trecvid.data.html)
[[12] WebVid10M] (https://m-bain.github.io/webvid-dataset/)
[[13] captions from @jeinsong] (https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/402146)
[[14] captions from @xiaozhouwang] (https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/398529)
[[15] DiffusionDB 2M 224x224 resized dataset from @atom1231] (https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/396136)
[[16] DiffusionDB 2M captions] (https://huggingface.co/datasets/poloclub/diffusiondb/tree/main)
