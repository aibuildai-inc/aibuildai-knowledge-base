# 18th Place Solution - Homemade Vectorizer + ImageReward + NLTK

Competition: drawing-with-llms
Rank: #18
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/582286

This is my first Kaggle competition. In my spare time, I work on a side project developing algorithms and tools for creating vector art. Therefore, I am thrilled to have found this competition, which gives me the opportunity to test my algorithms and explore the new possibilities brought by machine learning. I have learnt a lot from public codes and discussions. Thank you to the organizers for hosting the event and to all the participants for sharing so many wonderful ideas.

My approach is based on the following paper:

> Karthik, Shyamgopal, Karsten Roth, Massimiliano Mancini, and Zeynep Akata. "If at First You Don't Succeed, Try, Try Again: Faithful Diffusion-based Text-to-Image Generation by Selection." arXiv preprint arXiv:2305.13308 (2023).

This work claims that ranking generated images with [ImageReward](https://github.com/THUDM/ImageReward) will increase the TIFA score. The overview of my pipeline is:

1. Generate several images using Flux.1 Schnell
2. Vectorize the images using my own method
3. Calculate the ImageReward metric and the aesthetic score of each image
4. Pick the image with the highest score and do final adjustments (e.g. changing vectorization parameters, adding a letter for OCR)

During step 1 and 3, I also use [NLTK](https://www.nltk.org/) to alter the original prompt.

## Raster Image Generation

I deployed a quantized Flux.1 Schnell with the following prompt: `{image_description}, stylized proportions, exaggerated form, children illustration`. For quantization, I tried both Q4 and nf4. Q4 behaves slightly better.

## Vectorization

I took codes from my [another project](https://github.com/chsh2/nijiGPen/blob/main/operators/operator_io_raster.py) for image vectorization. It may not be as good as VTracer in general, but I tried to focus on generating clean geometry without unnecessary structures. Therefore, I think it may fit here.



The method is based on a common approach that first performs color quantization and then extracts the contours as polygon paths. While I tried to make some improvements in each step:

1. **Color Quantization**: To balance the efficiency and quality, I use a combination of K-means clustering and [Felsenszwalb algorithm](https://scikit-image.org/docs/0.25.x/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb). I also convert the image from RGB to LAB space for better result.
2. **Contour Finding**: Both OpenCV and Skimage provide `find_contours` API for a binary bitmap, so it needs to be called multiple times for each color in the quantized image. But the underlying algorithm (marching squares) can actually be applied to a multi-color bitmap without changes. I reimplemented this algorithm so that it only needs to be run once during vectorization.
3. **Path Simplification**: During simplifying each polygon path, I keep points which have 3 colors in their neighborhood unchanged. Otherwise, gaps may appear between adjacent shapes. This improvement is less important, since gaps do not necessarily lead to low aesthetic scores. I cancel this constraint when a higher level of compression is needed. The method to simplify a path is the Douglas-Peucker algorithm.



I prepared 4 sets of vectorization parameters that satisfies the SVG file size limit. Scores are estimated for each version of the converted image to determine the best one.

## Prompt Processing & Scoring

The paper I cited calculated the ImageReward metric using the original prompt. I discovered that dividing the prompt into several parts could result in a higher competition score. Using NLTK, I located all the nouns and created a sub-prompt for each one. Then, I used a concave function to get the average of the ImageReward metrics of all the sub-prompts.

In my tests, ImageReward behaves better than Clip and SigLIP as an indicator of the competition score. I did not compare it with VQA approaches though.



I also use NLTK for another purpose during image generation. I found that Flux.1 always failed to generate correct images of some abstract subjects such as trapezoids. I tried to alleviate this problem by attaching the word definition to the prompt. For example:

```plaintext
magenta trapezoids (a quadrilateral with two parallel sides) layered on a transluscent silver sheet (any broad thin expanse or surface)
```

This approach does not always make improvements. Therefore I generated multiple images for both versions of prompts for ImageReward to score.

## Other Thoughts

I am very grateful to Kaggle for providing GPU resources that have enabled me to put various ideas into practice. At the same time, I am also impressed by the organizers' ongoing efforts to improve the scoring metrics to ensure the fairness of the competition. 

However, I have one regret that my initial goal of testing my vector algorithms was not fully achieved. Besides raster image vectorization, I have some other algorithms for vector art, but cannot find a place to use them in the competition, since the metrics are quite specific to raster images. The aesthetic score is trained on photos, and VQA is also less precise for vector images. On the other hand, there is no metric evaluating vector image features or the SVG format. It is not a surprise to me that most participants prefer diffusers to LLM, since it looks more like a competition to generate highly-compressed raster images instead of vector images.

My takeaway from this is that we should perhaps be more aware of the limitations of this type of purely training-based metrics/scorers. This may be a bit off topic, but I do not like the fact that some people treat artists merely as training sources and downplay their expertise and judgements. On the one hand, as someone developing tools for artists, I am distressed to see the increasing conflicts between developers and artists. On the other hand, I also believe that respecting each other's knowledge is the key to make further progress.
