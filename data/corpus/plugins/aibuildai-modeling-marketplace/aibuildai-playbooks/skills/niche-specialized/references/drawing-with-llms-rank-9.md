# 9th place - SDXL LoRA fine-tune

Competition: drawing-with-llms
Rank: #9
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/581061

My general approach was:
- Generate images with a diffusion model
- Add an image frame and gradient background to increase aesthetic score
- VQA with basic questions to choose a candidate image
- Try slight variations to the image to boost aesthetic score further

Notebook: https://www.kaggle.com/code/linrock/9th-place-sdxl-lora-vqa-boost-aesthetics

### Example outputs:
Rotations in gradient direction and variations in frame color were meant to improve aesthetic score with minimal effect on VQA.



### Diffusion model config:
- SDXL base 1.0
- TinyVAE to speed up image generation
- LoRA trained on quantized-color images, filtered for high aesthetics and prompt alignment
- sampler: DPMSolver++ Karras 2M
- guidance scale: 5
- num inference steps: 8
- image size: 768x768

### LoRA training:

I generated 1.5k descriptions with LLMs, and images with FLUX dev and Hyper SDXL, using prompt variations to increase variety. Images were quantized to 24 colors, with median and gaussian filtering applied to remove details. Datasets were composed of the top scoring image for any given base prompt, based on paligemma2 VQA and aesthetic scores.

Example training images:


The goal was to steer image generation towards compositions and color palettes with high scores, given generic text prompts.

Trained with diffusers `train_text_to_image_lora_sdxl.py` with an offset noise of 0.0357 to enable high-contrast images that base SDXL is otherwise unable to generate.

This approach yielded better LB scores than all combinations I tried of base models (ie. SDXL, SD 1.5, Flux), community models and LoRAs, and prompting.


### Basic VQA questions:

- Does the image show `<prompt>`? yes / no
- Are there `<prompt>`? yes / no
- Does `<prompt>` exist in the image? yes / no

These were used both for VQA score filtering of LoRA datasets and ranking candidate images during image prediction. In submissions, I tried creating custom-tailored VQA questions out of prompts using Qwen, but didn’t find anything better than combos of basic questions.


### Misc:

- I used differential evolution to search for image modifications to improve aesthetic scores, and found that adding image frames on green backgrounds tended to do well
- Other than data generation, I didn't find any useful way to involve LLMs in the image generation process
- PNG to SVG code was incrementally improved upon Rich Olson’s notebook, and later on the OCR decoy, as it looked better than the signature I previously used
- The highest-aesthetic score post-blur images were consistently vividly-colored landscapes:

