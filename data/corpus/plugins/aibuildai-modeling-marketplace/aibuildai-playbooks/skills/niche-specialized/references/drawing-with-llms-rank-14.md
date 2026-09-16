# Best outputs solution?

Competition: drawing-with-llms
Rank: #14
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/581063

I've tried several solutions for this competition, but let's start with my selected submission.

### Fine-tuned Diffusion model + Vtracer + SVG optimization + multithreading (10 samples per description)...
First, I tried fine-tuning SSD both LoRA and full-finetuning, but I couldn't get good results. Then I tried  [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) an 8B model based on FLUX.1-schnell, and fine-tuned it on 8.5k rasterized svg images. I did a few iterations and got good results, but not good enough. And the model was big and slow.
Then I fine-tuned [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) a 2B DiT model like flux. After a few attempts, I started getting the results I wanted (Fully flat images with minimum details). However the model wasn't fast enough at 1024x1024 and it wasn't producing good results below 768x768. So I adapted the dataset and trained the model to generate from 256x256 and up.

I used multithreading (I've found a way to avoid the [multithreading issue](https://www.kaggle.com/competitions/drawing-with-llms/discussion/578877#3201519) in Kaggle packages) to run inference in parallel on the 2 GPUs and generate 5 images on each at 320x320, used VTracer to convert the image to SVG, and a Node.js local server to optimize the SVG. I repeat the Vtracer + NodeJs optimization iteratively until I get <10K SVG with the least deterioration possible. I then used [zhibinlan/LLaVE-2B](https://huggingface.co/zhibinlan/LLaVE-2B) to score prompt alignment and the aesthetic scorer from the metric code and pick the highest scoring output. I also applied a small circle as OCR decoy (from @richolson's notebook). Here are a few samples:















I tried many different methods like generating vqa on the fly and using the metric code to score, removing the OCR decoy, (which affect the aesthetic score) for images that score 1 for OCR, for images that score <0.99, place  the decoy at different corners and get the highest scoring images.... But none helped.
I also tried a fine-tuned Flux-dev which produced excellent results, better than lumina (the images above), but for some reason I got a lower score 🤷.

Here is the [notebook](https://www.kaggle.com/code/sitaberete/new-fork-of-last-730-score-drawing-svg-with-sv?scriptVersionId=242050046), but the code is super messy and you will notice that all the methods are defined in the __init__ method to avoid the multithreading issue I mentioned above.

### Qwen 2.5 VL 3B
The first solution I tried, and honestly, the one I had the most fun with, was fine-tuning qwen-2.5-VL-3B model. I started by a 2M+ samples dataset from [starvector/text2svg-stack](https://huggingface.co/datasets/starvector/text2svg-stack) and spent a few days cleaning, simplifying, and formating the SVG using a specific format that should facilitate learning for LLMs. But the dataset was too noisy and I didn't have the time, resources, and experience to efficiently filter it. And the majority of the SVG where icons and black and white with horrible aesthetic. So I collected a dataset of 8.5k SVG with a decent aesthetic level and that are 10Kb or less (These are the SVGs I rasterized to train the diffusion models). I spent nearly 2 weeks trying to fine-tune it but couldn't get good results. The model was too small from my understanding, at the beginning of the training, see improvements at each checkpoint, but later on it starts to degradate always around the same step count.
I had the intention when the model learn to generate good enough SVGs to fine-tune it again to teach it to "think" with GRPO so that when the initial LLM generates an initial SVG I will feed it to the thinking model to analyze it and refine it iteratively withing it thinking process until it get a good result.

Although the VLLM approach didn't work, it was what made joining this competition worth it for me, because for the diffusion solution, I spent more time trying to find a way around the issues of the metrics than improving the model.

### About the Metric code
It turned out that the issue isn't only the OCR, after I got stuck at a score of 730 in the public LB, I intentionally decreased the quality of my outputs by aggressively optimizing them to the point where the shapes are almost not noticeable, only the colors where clearly visible, but still got similar scores locally and for some samples even higher score despite the OCR score being 1.0 for all of them.

Anyways, congrats to all the winners, enjoy it!
Also, congrats to @ryanholbrook for making the most mediocre competition metrics and sticking to it despite being reported by many (some almost 2 months ago).
