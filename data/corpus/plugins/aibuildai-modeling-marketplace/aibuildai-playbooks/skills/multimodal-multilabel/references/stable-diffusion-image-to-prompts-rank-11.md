# 11th Place Solution

Competition: stable-diffusion-image-to-prompts
Rank: #11
Source: https://www.kaggle.com/c/stable-diffusion-image-to-prompts/discussion/410611

Thanks to the Kaggle staff and host for organizing the competition and congratulations to the winners. I appreciate my teammates @charmq and @yoichi7yamakawa!

## Overview
We took the approach of training the image model to directly predict the target embeddings. For almost the entire duration of the competition, we used the image encoder of OpenCLIP-ViT/H as a backbone, which is a counterpart of the text encoder used in Stable Diffusion v2. In this competition, it was much stronger than usual ImageNet pre-trained models, such as EfficientNet or ConvNeXt. However, we realized that OpenCLIP-ViT/bigG and BLIP-2 worked better in the last 1-2 days, which we didn't have enough time for tuning and training fully. Our final submission was an ensemble of OpenCLIP-ViT/H, OpenCLIP-ViT/g, OpenCLIP-ViT/bigG, and BLIP-2.

## Dataset
We downloaded and generated 4.5M+ images in total and used a part of them as a validation set. The validation scores correlated a lot with LB scores, although they were much higher (0.7+). The datasets we used are as follows. We extracted the prompts with cosine similarity away from each other. Also, we randomly concatenated [modifiers from Open Prompts](https://github.com/krea-ai/open-prompts/tree/84d1628e7b3d97a45d237311579faf39692f2a02/modifiers) to a part of the dataset.
- [DiffusionDB](https://poloclub.github.io/diffusiondb/)
- [MagicPrompt-Stable-Diffusion](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion)
- [Microsoft COCO captions](https://cocodataset.org/#home)
- [gpt_generated, hardcoded (Kaggle)](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/398529)
- [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/)
- [nocaps](https://nocaps.org/)
- [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)

We first used the [official script](https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/scripts/txt2img.py) and mainly used the Diffusers library to generate images. The default parameter of `guidance_scale` differed from the official script, but we used the default parameters since it did not affect the performance a lot.

## What worked
The backbones we used can only get images of (224, 224). We split a (448, 448) image into 4x (224, 224) patches and concat each patch (+full image) embeddings. This improved the performance but slowed down the speed of training and inference by 5x times, so we used it with normal models with (224, 224) input.

The following small techniques worked a lot to fine-tune large models.
- freezing of first layers
- layer-wise learning rate decay
- pre-trained weight decay

Also, contrastive loss using text encoders worked for CLIP models.

## What did not work
- Make use of Stable Diffusion's weight, such as encoder, decoder, unet
- Training from Stable Diffusion's latent
- Image2text
- Auxiliary loss, such as CLIP text embeddings or prompt prediction
- Decrease the step size in image generation (we thought it might work as data augmentation)
- knn with trained models
- CLIP Interrogator

## Acknowledgement
We deeply acknowledge great OSS such as PyTorch, PyTorch Lightning, Hugging Face, etc. We would also like to appreciate Preferred Networks, Inc for allowing us to use computational resources.
