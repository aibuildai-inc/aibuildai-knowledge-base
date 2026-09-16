# 26th Place Solution: SDXL Flash + Optimized img2svg + TIFA QA Generation

Competition: drawing-with-llms
Rank: #26
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/581069

[Sample output]
I’d like to extend my sincere thanks to the organizers for their hard work and dedication in making this competition both engaging and meaningful. I also appreciate everyone who shared helpful ideas throughout—this community support made a big difference. My custom image-to-SVG pipeline was inspired by the work of @richolson , and I’m especially grateful for his contributions.

I'm a beginner in this field with limited experience and knowledge, so feedback and suggestions are always welcome!

# TL;DR
- Fine-tuned **SmolLM2-1.7B-Instruct** to generate four TIFA-style questions per description.
- Used **SDXL Flash** for fast and high-quality image generation.
- Converted images to SVG using two approaches: a custom polygon/path pipeline and VTracer with binary search for size optimization.
- Injected a small **"L" watermark** in each SVG to mitigate OCR hallucination.
- Used **PaliGemma-3B** (about 3x faster than the 10B version) to evaluate and select the best SVG based on competition score using my generated TIFA questions.
- Implemented a **dynamic time management** strategy to maximize usage of the per-prompt time budget.
- Here is the **[inference notebook](https://www.kaggle.com/code/kawchar85/sdxl-flash-img-svg)**

# TIFA Question Generation
To generate high-quality evaluation questions, I created a dataset of approximately 10,000 structured examples using Gemini and Claude. Each set follows a strict 4-question TIFA format:

- Q1: Yes/No question that contradicts the description (answer: “no”)
- Q2: Multiple-choice question using an exact word from the description
- Q3: Another multiple-choice question using a different exact word
- Q4: Yes/No question affirming something present in the description (answer: “yes”)

I fine-tuned SmolLM2-1.7B-Instruct on this dataset. The model performs well and significantly improved my competition score. In around 2% of cases, Q2 and Q3 were duplicates—likely due to minor inconsistencies in my training data or the limited capacity of the small model. To handle such edge cases, I added fallback logic to inject dummy questions when fewer than four unique ones are generated.

Inference is extremely fast, averaging about **3 seconds per prompt**.

# Image Generation
I used SDXL Flash due to its speed-quality tradeoff. Prompts were constructed using a consistent template:
>flat vector illustration of {description}, minimalist design, solid colors, flat color blocks

Each prompt gets 4 attempts by default, extended dynamically using leftover time from previous predictions.

# Image-to-SVG Conversion
Each generated image is first resized from 768×768 to 384×384 to reduce SVG complexity. The image is then quantized to 16 colors using MiniBatchKMeans in LAB color space. After quantization, I apply a **Disjoint Set Union (DSU)** algorithm to merge adjacent regions with similar colors. The processed image is then passed to two SVG conversion pipelines:

### Custom Polygon/Path Simplification
This method allocates a total point budget (e.g., 2000 points) and distributes it across contours based on an importance metric combining area, shape complexity, and proximity to the image center. For each contour, I perform a **binary search** to find an optimal epsilon value that simplifies the contour to the target number of points using cv2.approxPolyDP.

Each simplified shape is then encoded as either a polygon or a path, depending on which representation is shorter in bytes. This yields visually accurate SVGs while keeping file size under control.

### VTracer + Binary Search Compression
As an alternative, I also use VTracer to convert the image into layered SVG paths. After tracing, a **binary search** simplification pass is applied to the path elements using cv2.approxPolyDP to further reduce the file size and ensure the final SVG remains under the 10KB limit.


For each generated image, I produce two SVGs—one using the custom polygon/path pipeline and the other using VTracer with binary search.

# OCR Hallucination Defense
The competition's scoring model often hallucinates the presence of text in SVGs, negatively impacting the OCR score and overall performance. To counter this, I inject a small "L" shaped watermark into a fixed position within each SVG.

The color of this watermark is chosen dynamically—either black or white—based on the local brightness of the region to ensure it blends naturally with the image while still being detectable. This simple yet effective trick significantly reduces false positives in OCR evaluation.

# Best SVG Selection
To select the best SVG, I use **PaliGemma-3B**, which is approximately **3× faster** than the 10B version, with minimal loss in accuracy.

# Time Management Strategy
Each description is allocated a fixed time budget of 64 seconds. If a sample finishes early, the unused time is accumulated and carried forward.

This saved time is then reused to allow additional image generation attempts for future descriptions, calculated as:
`attempts = int(base_num_attempts + saved_time / per_attempt_time)`

# Submission Selection
To guide submission selecting, I created a validation dataset of 100 descriptions covering **11 diverse categories** (Nature, Abstract, Fashion, Animal, Household, Food, Urban, Technology, Transport, Sports/Recreation, Medical/Healthcare), all generated using Gemini.

You can find the dataset [here](https://www.kaggle.com/datasets/kawchar85/t2s-validation-dataset).

I ran multiple inference notebooks and tracked both CV and leaderboard performance:
| Notebook | Public LB | CV Score | Private LB | Final Submission |
|----------|-----------|----------|------------|------------------|
| 1        | 0.699     | 0.724    | 0.690      | No               |
| 2        | 0.696     | 0.730    | 0.699      | No               |
| 3        | 0.690     | 0.752    | 0.686      | Yes              |
| 4        | 0.689     | 0.763    | 0.693      | Yes              |



Notebook 3 and Notebook 4 were selected for final submission based on strong cross-validation performance. Unfortunately, I missed selecting Notebook 2—which actually had a slightly higher private leaderboard score.

# Other Approaches I Tried
- Combined Qwen2.5-14B-Instruct-AWQ with SDXL Hyper, but this setup led to lower scores overall.
- Experimented with LoRA fine-tuning on SD, SDXL, and SDXL Flash models, but this consistently reduced performance.
- Fine-tuned SmolLM2-1.7B-Instruct for prompt enhancement in an SVG-friendly style. This didn't improve the public leaderboard score (actually reduced it to 0.682), but interestingly achieved a slightly better private score of 0.694 (compared to 0.693 in my selected submission).
- Tried various diffusion models and found that SDXL Flash, Flux Schnell, Hyper SDXL, Juggernaut XL V9, and Playground 1K Aesthetic tended to perform better.
- Attempted to dynamically calculate the total point budget for SVG simplification based on image complexity, rather than using a fixed 2000-point budget. However, this added overhead without significantly improving results.

# Mistakes I Made
Looking back, there are a few key areas where I could have improved:
- **I didn't make any focused attempts to optimize the aesthetic score**, which could have boosted my overall competition metric.
- I relied too heavily on a single dataset for cross-validation and final submission decisions, which likely led me to overlook better-performing solutions (e.g., Notebook 2).
- I avoided exploring more unconventional or "hacky" techniques like those used by the top-ranking teams.
