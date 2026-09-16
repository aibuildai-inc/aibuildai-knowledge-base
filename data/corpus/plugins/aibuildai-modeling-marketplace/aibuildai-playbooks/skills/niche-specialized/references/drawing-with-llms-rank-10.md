# 10th Place Solution: Flux 1 Dev -> SVG optimization -> Postprocessing

Competition: drawing-with-llms
Rank: #10
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/581213

# 10th Place Solution: Drawing with LLMs

**Team:** BERA

We’re thrilled to have placed 10th in the "Drawing with LLMs" competition. Thanks to the organizers and metric team for continuously iterating and improving the challenge and other participants for sharing great work. 

Our solution focuses on **generating high-quality images first**, and then **converting and optimizing them into compact, accurate SVGs** under the tight 10KB constraint. Our pipeline emphasizes image quality, conversion fidelity, and smart selection via the competition's evaluators, the efficiency of the SVG conversion being particularly strong.

## Pipeline Components
Image (Flux-1-Dev + HyperSD + Nunchaku) + SVG (vtracer + svgo + custom geometry simplification + OCR addition) + Selection (VQA + Aesthetic + OCR) + Optional Border Addition

## Overview of Our Pipeline

1. **Image Generation (Flux 1 Dev + HyperSD + Nunchaku)**

   * We used **Flux 1 Dev**, enhanced with **HyperSD LoRA** to reduce inference to **8 steps** from default **50 steps** and **Nunchaku** for loading the transformer part in a much faster version. We quantized all components of the pipeline in **4bit** with BitsAndBytes so that they fit on a **single T4**.
   * Initially, we couldn't get parallel generation to run, but switching to the newest Nunchaku dev version resolved the issue.
   * Final throughput: \~10 images in \~20 seconds using a single T4.

2. **Image → SVG Conversion (vtracer + Path Simplification)**

   * Each image was converted using **vtracer**, which gave us clean, layered SVGs.
   * We applied **geometric simplification** by trimming intermediate points that added negligible curvature:
     > e.g., If point B lies within 2 pixels of line A–C, we drop B.
* Compression with **SVGO** followed. We performed **binary search over vtracer parameters** to find the best configuration that used under 10000 bytes.[
]

* This step was expensive and ran **multiprocessed on 4 threads**.
* We also added  **two small “M” watermark** in black and white to mitigate OCR false positives.


3. **Candidate Selection (VQA + Aesthetic Score + OCR Robustness)**

   * SVGs were scored using the **VQA and Aesthetic evaluators**.
   * We ranked all candidates, then searched *backwards* to find the first image that **passed OCR checks** (a major pain point).
   * On the top candidate, we created **two variants**: one original, and one with a **gradient silver border**. We used the best-scoring one.

[Example Output with Gradient Silver Border]

## Debugging + Engineering Pain Points

We spent **dozens of hours debugging edge cases**, and the new submission format introduced several non-trivial issues:

* **SVGO**: Wrapped with a custom Python interface and loaded it into a private GitHub repo, then installed it in the dependency manager with PAT (personal acess token). 

* **Nunchaku**: Didn’t work via standard install—had to fork and install via GitHub and then still load in the binary manually.

* **Multiprocessing and Execution Graph Dependencies**: Ensuring image generation on cuda:0, SVG conversion and cuda:1 image selection stages didn’t block each other would have been a huge speed improvement but we could not get it to run on the submission server, though it ran in notebooks, on save and even on importing the package.  

## Notes on Generation Behavior

* **Landscapes**: Near-perfect every time. The best modality for our pipeline.
* **Fashion**: Challenging—details often lost in SVG conversion due to subtle features.
* **Geometry**: Almost hopeless. Across 12+ models, even basic shapes like trapezoids only succeeded \~10% of the time.

## Alternative Submission (Not Selected)

We had another submission that scored much higher on **private LB**, but slightly worse on **public**, so we sadly didn't select it, but maybe it's still interesting for you to hear of these additional improvements:

* It involved **prompt rewriting**, passing different forms to different encoders.
* Initially failed because we sent the **same format** to both:

  * CLIP prefers **comma-separated keywords**.
  * T5 prefers **natural sentences**.
* Once separated, prompt rewriting improved performance in niche cases (e.g., animals and food).

## Final Thoughts

We learned a lot about image generation pipelines and SVG compression. The challenge pushed us to get creative with pipelines, build interpretable evaluators, and respect tight compute budgets.

Thanks again to the hosts and congrats to all participan
