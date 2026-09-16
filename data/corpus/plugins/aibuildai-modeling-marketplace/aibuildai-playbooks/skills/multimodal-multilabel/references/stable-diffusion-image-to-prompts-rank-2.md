# 2nd place solution

Competition: stable-diffusion-image-to-prompts
Rank: #2
Source: https://www.kaggle.com/c/stable-diffusion-image-to-prompts/discussion/410606

Congrats to all the winners, and thanks to kaggle for hosting such an interesting competition. This competition was a great learning experience for me. I am really looking forward to reading everyone's solutions.

My solution is basically the same as the ViT-based method in the public notebook. I created my own dataset by running Stable Diffusion and trained a model to predict sentence embeddings in a supervised manner. The overview is shown in the figure below.

[image]

## Dataset Generation

### Preparation

The following modifications were made to 4x speed up the generation with fp16.

- Change the scheduler from DDIM to DPMSolver++ (diffusers.DPMSolverMultistepScheduler) and change the number of steps from 50 to 16.
- Use xformers

There was concern that reducing the number of generation steps would cause a shift in the mapping from prompt to image, but there was no significant qualitative difference, and the advantage of increased generation speed was judged to be greater, so I chose this method.

The xformers led to a speedup of about 30%. xformers were also used in the training of the model, which is described after.

Note that the random numbers used in the generation have a significant impact on the overall image generated, so it is necessary to generate images using different seeds. At first I fixed the seed to 0 to ensure reproducibility, but I realized this mistake and used a different seed for each prompt, which greatly increased the score. I was generating the seed using zlib.adler32, the name of the prompt set, and the number of the prompt.

As for the guidance scale, I had been using the default value of 7.5 from diffusers until I realized that 9.0 seemed to be the correct value ([link](https://github.com/Stability-AI/stablediffusion/blob/main/scripts/txt2img.py#L142)), so I used data generated with either value. LB scores showed that there was no significant difference between 7.5 and 9.0, but scores with 3.0 were significantly lower.

### DiffusionDB

To deal with the vocabulary specific to Stable Diffusion, we generated images using prompts from [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb). Excluding duplicates, there are approximately 1.8M Prompts, and images were generated for all of them.

### COCO Captions

Captions from [Microsoft COCO Captions](https://arxiv.org/abs/1504.00325) were used to generate the images. 500k images were generated from approximately 600k captions in train and validation.

### Open Images

To create even more diverse prompt sets, I utilized the natural images in [Open Images Dataset V3](https://github.com/openimages/dataset/blob/main/READMEV3.md) (OID) and Image-To-Text pretrained models. Images from OID are input to the Image-To-Text model, and the resulting caption is input to StableDiffusion as a prompt. OID contains 9M natural images, of which the first 5M were used to generate.

[Salesforce/blip2-flan-t5-xxl](https://huggingface.co/Salesforce/blip2-flan-t5-xxl) was used as the Image-To-Text model. I used Nucleus sampling with `do_sample=True` and `top_p=0.9`.

The sample prompts in the competition consist of multiple objects and multiple sentences, but captions output by BLIP-2 tends to produce very concise sentences by default. Therefore, by chaining BLIP-2's caption generation, I made sure that sentences with a certain level of complexity are generated. The procedure for chaining is as follows
1. generate a sentence following `"a photo of"` as usual (denote the generated sentence as A)
2. generate a sentence following `"a photo of" + A` (denote this as B)
3. Output `A + B` as the final caption

The Prompt set created is as follows.

|Prompt set name|Subset|# of images|Chain|
|---:|---:|---:|---:|
|OID1|train0|1M|2|
|OID2|train1|1M|3|
|OID3|train2|1M|1*|
|OID4|train3|1M|2|
|OID5|train4|1M|2|

For OID3, in order to realize a combination of objects that do not exist in natural images, prompts were created by connecting two short captions created with chain=1 with commas. In concrete terms, the short captions created in chain=1 were converted to sentence embeddings using SentenceTransformer, and divided into 1000 clusters using Spherical KMeans. Then, for every combination of two clusters, we randomly selected one caption for each cluster and connected them with a comma.

### ChatGPT (gpt-3.5-turbo)

This is a dataset that has utilized the ideas from [this notebook](https://www.kaggle.com/code/safavieh/chatgpt-generated-prompts/notebook). When I tried generating prompts using ChatGPT, I found that using the same prompt only yielded similar generation results. Altering parameters such as temperature did somewhat improve the situation, but it also made it more likely to produce broken sentences or excessively long ones. Therefore, I considered modifying the prompt sent to ChatGPT utilizing the previous generated prompts, and implemented it with the following code.

```python
command = (
    "Describe a fairly random scene with multiple objects by one short sentence with some modifiers in one line."
    "\nHere are some examples of such texts:"
)
sample_set = [
    "hyper realistic photo of very friendly and dystopian crater",
    "ramen carved out of fractal rose ebony, in the style of hudson river school",
    "ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus",
    "a thundering retro robot crane inks on parchment with a droopy french bulldog",
    "portrait painting of a shimmering greek hero, next to a loud frill-necked lizard",
    "an astronaut standing on a engaging white rose, in the midst of by ivory cherry blossoms",
    'Kaggle employee Phil at a donut shop ordering all the best donuts, with a speech bubble that proclaims ""Donuts. It\'s what\'s for dinner!""',
]
generated_prompts: Sequence[str] = []
for _ in range(num_trials):
    if len(generated_prompts) < 5:
        examples = list(sample_set)
    else:
        examples = list(np.random.choice(sample_set, 4, replace=False)) + list(
            np.random.choice(generated_prompts, 3, replace=False)
        )

    np.random.shuffle(examples)
    prompt = command + "".join([f'\n"{e}"' for e in examples])
    generated_prompts = send_to_chat_gpt_and_cleansing(prompt)
```

In this method, I created about 1M prompts, inputted them into Stable Diffusion after removing duplicates, and produced about 1M images.

I also tested a few other LLMs, especially focusing on those with open weights, but I was unable to obtain a prompt set equivalent or superior to gpt-3.5-turbo using the same method, based on my qualitative evaluation. Typical behaviors of such LLMs are starting some sort of explanation instead of answering in the desired format or only being able to produce slightly modified examples. I have not tested LLaMA and its derivative models because I was unable to obtain LLaMA's weights in a license-compliant manner.

## Model and Training

### CLIP Models

In the initial, simplistic experiments, the pretrained models with CLIP performed better than those for ImageNet classification. As a result, I focused on fine-tuning CLIP models.

The following four models were used in the final submission.

 - ConvNeXt xxlarge ([CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind))
 - BLIP-2 VisionModel (EVA01-g?, [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b))
 - EVA02-L ([timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k](https://huggingface.co/timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k))
 - EVA02-e ([timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k](https://huggingface.co/timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k))

### Linear-Probing

Regarding the fine-tuning of CLIP, it has been suggested that Linear-Probing (LP), which trains only the last linear layer, and [LP-FT](https://arxiv.org/abs/2202.10054), which fine-tunes the entire model after LP, are effective to generalize. Indeed, these methods were effective in my experiments as well.

### Resolution

Increasing the resolution also had a significant effect. CLIP models are typically trained at a resolution of 224, but I believe that 224 was disadvantageous for recognizing complex contexts containing multiple objects.

ViT can infer with only a slight degradation in performance for inputs at a higher resolution than during training, by interpolating positional encoding. For instance, you can use functions like [resample_abs_pos_embed](https://github.com/huggingface/pytorch-image-models/blob/v0.9.2/timm/layers/pos_embed.py#L17) implemented in timm. By using this, you can use a model trained at a lower resolution as a good initial value when training at a higher resolution.

Until the middle of the competition, I was using models trained at 224, but towards the end, I improved my score by using those as initial values and training models at 336 or 448.

### Q-former

SentenceTransformer's sentence embeddings are the normalized sum of token embeddings. I considered leveraging these token embeddings as additional supervision signals.

I implemented an architecture using the Q-former from BLIP-2 to convert the patch embeddings output by the vision model into token embeddings with a length of 77. I then compared the output of that model to the token embeddings from the SentenceTransformer, padded with zeros until they reached a length of 77, and trained the model based on this comparison. Simply taking the mean squared error (MSE) would be heavily influenced by word order, so I used the Hungarian method to find the matching with the minimum MSE and calculated the MSE between those matches. I also used the cosine similarity between the sentence embeddings.

Simultaneous training of the Q-former and the vision model was unstable, and I could not achieve success during the competition period. However, by freezing the pretrained vision model and training only the added Q-former, I achieved a performance improvement of +0.001 to +0.003 on the leaderboard (LB) for individual models. Moreover, since the vision model is frozen, I could obtain its standalone output of the vision model with almost no additional cost when predicting with the Q-former. It provided a slight benefit in ensembling.

The effectiveness of the Q-former was not particularly pronounced in this problem setting, but I believe it would be useful in the original sense of an inversion task, where the goal is to reproduce images.

### Weighted Averaging

For ensembling, I used weighted averaging. The weights were optimized using Adam and validation set from my ChatGPT dataset.
