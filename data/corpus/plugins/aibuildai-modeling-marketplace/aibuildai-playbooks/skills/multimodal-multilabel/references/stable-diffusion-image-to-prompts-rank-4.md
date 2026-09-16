# 4th Place Solution

Competition: stable-diffusion-image-to-prompts
Rank: #4
Source: https://www.kaggle.com/c/stable-diffusion-image-to-prompts/discussion/410798

Congratulations to all the winners! Thank you, Kaggle, for hosting such a fun competition (kudos for coming up with 16K prompts)! Also, a shoutout to all the open-source efforts in the DS community.

[Final Submission Notebook](https://www.kaggle.com/code/gerwynng/sd-itp-4th-place-solution)
# Summary

|  | Public | Private | Runtime |
| --- | --- | -- |
| (1+2+3) Text Candidate Generation |  0.614X | 0.612X  | 6 hours|
| (4) ViT Encoders| 0.644X | 0.641X | 3 hours |
| Weighted ensemble (CV Best: 0.4, 0.6) | 0.646X | 0.643X | 9 hours |
| Weighted ensemble (LB Best: 0.25, 0.75) | 0.648X | 0.645X |  9 hours |

**Text Candidate Generation**
- Text retrieval: performing a top-k exhaustive search from 56M precomputed text embeddings, using pretrained [CLIP-bigG](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k), followed by cosine similarity filtering using pretrained [CLIP-H14](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
- CLIP interrogator: performing a top-k exhaustive search from precomputed CLIP-bigG text embeddings of 1.6M "prompt components". Each of the retrieved string components are concatenated into prompts.
- CoCa: using [LAION's implementation](https://github.com/mlfoundations/open_clip#fine-tuning-coca). 

**ViT Encoders**
Two-stage finetune process using: 1. DDB; 2. generated 7M images
Initialised from the following checkpoints: CLIP-ViT [L](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K), H (as above), [G](https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s12B-b42K), bigG (as above)

# Dataset 
**Caption Set (Text Retrieval + Image Generation)**
Total: ~5.2M, Generated Images: ~5.2M 
- [DDB] (https://github.com/poloclub/diffusiondb)
- [Gustavosta's Stable Diffusion Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)
- [Kree AI's Open Prompts](https://github.com/krea-ai/open-prompts)
- [Stable Diffusion Discord Prompts](https://huggingface.co/datasets/bartman081523/stable-diffusion-discord-prompts)
- [Midjourney prompts](https://huggingface.co/datasets/succinctly/midjourney-prompts)
- [tanreinama's 900k](https://www.kaggle.com/datasets//900k-diffusion-prompts-dataset)
- [Google Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/)
- [nocaps](https://nocaps.org/download)
- [coco](https://cocodataset.org/#download)
- [textvqa textcaps](https://textvqa.org/textcaps/dataset/)

**[WIT (Text Retrieval + Image Generation)](https://github.com/google-research-datasets/wit)**
Total: ~2.5M, Generated Images: ~1M

**Generated Prompts (Text Retrieval + Image Generation)**
Total: ~10M+, Generated Images: ~1M
- [leonidkulyk's magic prompt 1M](https://www.kaggle.com/datasets/leonidkulyk/magic-prompt-1m)
- xiaozhouwang's [sd2gpt](https://www.kaggle.com/datasets/xiaozhouwang/sd2gpt2) and [sd2hardcode](https://www.kaggle.com/datasets/xiaozhouwang/sd2hardcode)

i also generated more using [text2image-prompt-generator](https://huggingface.co/succinctly/text2image-prompt-generator) and [MagicPrompt](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion)

**[YFCC 100M open subset](https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset) (Text Retrieval only)**
Total: ~4M

**[Laion CoCo](https://huggingface.co/datasets/laion/laion-coco) (Text Retrieval only)**
Total: 27M (random sampled)

**[Datacomp Small](https://huggingface.co/datasets/mlfoundations/datacomp_pools/tree/main/small) (Text Retrieval only)**
Total: 5M

**Prompt Components (Interrogator only)**
Total: 1.6M components
- [English PoS word list](http://www.ashley-bovan.co.uk/words/partsofspeech.html)
- [Open Image Class List](https://storage.googleapis.com/openimages/web/download_v7.html)
- [CuPL Imagenet Prefixes](https://www.kaggle.com/datasets/gerwynng/img2text-prompt-components)
- [Original Clip Interrogator Components](https://github.com/pharmapsychotic/clip-interrogator/tree/main/clip_interrogator/data)
- + i took top frequent phrases (e.g. split prompts by `,`) from OpenPrompts dataset (~1.2M)

## Text Preprocessing 
Most stable-diffusion prompts are really messy with lots of unreadable/unnecessary words and punctuations. I manually looked at samples and tried to clean them. ([functions in this notebook](https://www.kaggle.com/gerwynng/sdi2p-text-preprocessing))
- removing near-duplicates (using sentence transformers' cosine similarity) works well for text-retrieval solutions; reduces 10-20% in dataset sizes while achieving similar scores. 

## CV
32k sampled prompts from Caption Set. Total is ~34k after allowing some prompts to repeat with different generated images.

# Solutions and Training
## Text Retrieval
No training was performed in this step.

I selected the top 60 candidates by calculating the cosine similarity between the image CLIP-bigG embeddings and my precomputed text embeddings. Then, for each of these top 60 candidates, I recalculated the cosine similarity using CLIP-H14 and filtered out those with a cosine similarity <= 0.29.

To perform exhaustive search on the 56M embeddings at inference time, I employed the following technique:
- I divided my dataset into different sections as listed above (dataset chunking).
- PCA with dimensions of 768 for all sections, except for Laion CoCo where I used 512 dimensions.
- mem-mapped numpy arrays in float16 format.
- a two-layer chunking approach. The outer loop involved splitting the 16k test image embeddings into 768 chunks, while the inner loop involved chunking the text image embeddings into 1024*1024 chunks.
- `torch.mm` calculations on GPU to compute the cosine similarity.

## CLIP Interrogator
I came out with 14 templates of how to create prompts from components. E.g. ``['adjectives', 'nouns', 'verbs', 'openprompts_modifiers', 'adverbs', 'movements']``.  For each template, I selected the top 8 candidates from each component, resulting in a total of 112 text candidates.

Initially, I experimented with random search, using permutations of prompt components + the number of components. However, I observed that this approach led to overfitting on the CV set too quickly. Thus, I decided to use the 14 templates that I manually selected from EDA.

## CoCa Model
This is low impact. I tried finetuning it, but it did not offer significant improvement over the pretrained version.

## Vit-Encoders
I kept batch size ~160, and then un-freezed layers depending on model size
- L: un-freeze from 2 onwards
- H: un-freeze from 14 onwards 
- G: un-freeze from 24 onwards 
- bigG: un-freeze from 36 onwards
 
Stage 1: DDB (~1.9M left after text-preprocessing, used their original images)
Stage 2: Self generated 7M
Both using LR: 4/5e-5 for 3/4 Epoch, using CosineEmbeddingLoss

Did not work: adding dropout (/ spatial), increasing FC layers, epoch ensemble, changing patch size

### TTA
In my final effort, I was able to achieve a 0.02X boost by utilizing different image mean and standard deviation normalization as Test-Time Augmentation (TTA). I used the pretrained statistics along with statistics computed from my 32K test set ([notebook on how i computed here] (https://www.kaggle.com/gerwynng/estimate-image-stats)). I performed two inferences and averaged the results, followed by a simple average across the four ViT encoders.

## Weighted Average (Text Candidates + ViT Encoders)
- Prior to taking a weighted average, I normalized the ST embeddings (from Text Candidates) and ViT embeddings.
- My CV only relied on one fold. As a risk mitigation strategy, I made one submission with weights based on the maximum CV (Text Candidates: 0.4, ViTs: 0.6), and another submission with weights based on the maximum LB score (Text Candidates: 0.25, ViTs: 0.75).
