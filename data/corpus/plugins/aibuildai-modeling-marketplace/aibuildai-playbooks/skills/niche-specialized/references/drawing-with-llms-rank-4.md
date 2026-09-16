# 4th Place Solution: SD3.5M + DRaFT + diffvg

Competition: drawing-with-llms
Rank: #4
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/581108

First, I’d like to thank the organizers for their hard work in running this competition! I’d also like to thank @richolson for sharing two extremely helpful bits of code: his [OCR decoy trick](https://www.kaggle.com/code/richolson/let-s-defeat-ocr-easy-lb-boost) and [image to SVG conversion](https://www.kaggle.com/code/richolson/stable-diffusion-svg-scoring-metric). 

## Overall approach

My pipeline first generates images with Stable Diffusion 3.5 Medium, fine-tuned with [DRaFT](https://arxiv.org/abs/2309.17400) to produce high-quality, score-optimized outputs. The images are then heuristically converted to rough SVGs, which are further refined using [diffvg](https://github.com/BachiLi/diffvg), a differentiable rasterizer.


## Diffusion Model
After experimenting with a few models, I decided to use Stable Diffusion 3.5 Medium without the T5 text encoder. It has a relatively small memory footprint, which is important for fine-tuning. 

## DRaFT Fine-Tuning

I trained a LoRA for SD3.5M with [DRaFT](https://arxiv.org/abs/2309.17400) (more precisely, DRaFT-LV), which trains the diffusion model to generate high-scoring images according to reward models. Unlike RL/DPO, DRaFT takes advantage of the fact that diffusion models are differentiable and backpropagates the reward directly into the diffusion model. You can see the training code [here](https://github.com/clarkkev/svg-diffusion)



**Training data:** I generated a set of 1000 training prompts with corresponding questions/answers using GPT-4o. I took a two stage-approach of first generating 50 image categories and then sampling prompts from those categories, with the known evaluation set categories (landscapes, abstract, and fashion) producing more prompts. 

**Reward functions:** I used two general reward functions that I knew worked well from my previous work on DRaFT: 
- [Human Preference Score v2](https://github.com/tgxs002/HPSv2)
- [PickScore](https://github.com/yuvalkirstain/PickScore) 

These models were trained on human preference data. I also added three competition-specific rewards:
- The competition aesthetic score
- A proxy for the VQA score (yes/no on “Does the image show {prompt}?”). This was faster and surprisingly seemed to work better than using the actual set of questions for each image. 
- Negative [LPIPs](https://github.com/richzhang/PerceptualSimilarity) distance between the image and convert_to_svg(image), so the model generates images that are easy to convert. 

**Results:** The fine-tuning improves aesthetic score by ~10% and VQA score by about ~5%. The overall LB score went up by around 0.03. A side benefit of the training is that I didn’t need to do any prompt engineering; I instead just relied on the fine-tuning to lead the model to produce the right image style. Here are some examples of generations before/after the fine-tuning (using the same random seed):



## SVG Conversion
I used a simple heuristic image-to-SVG converter inspired by Rich Olson’s notebook, with several tweaks to improve both visual quality and compression efficiency. The resulting SVG is then optimized with [diffvg](https://github.com/BachiLi/diffvg), a differentiable SVG rasterizer. The optimization minimizes the L1 distance between the rendered SVG and the original diffusion image using gradient descent on the locations/colors of the paths. This worked amazingly well, and meant I could get away with a pretty basic initial SVG converter, as the refinement would fix most issues that arose. I also used Rich Olson’s OCR decoy trick to reduce OCR hallucinations. 

## Best of n
For each prompt, my notebook keeps generating SVGs until roughly 60 seconds have passed, with some adaptivity based on whether it is behind or ahead of schedule. This typically produces 5-6 candidates. The SVG with the highest VQA score given the yes/no question “Does this image show {prompt}?” is returned. There is also a check to make sure the returned image has an OCR score of 1.
