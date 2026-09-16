# 16th place solution - SDXL-Flash + Siglip/AES Reranking + Proxy VQA

Competition: drawing-with-llms
Rank: #16
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/581094

First of all, we would like to thank the hosts for their continued effort to improve the metric to ensure that the top solutions aligns with the objectives of the competition!

Team members:
- @yeoyunsianggeremie - solution development and experimentation
- @xyzdivergence - solution development and experimentation
- @evgeniimaslov2 - dataset generation

Code: https://github.com/bogoconic1/16th-place-kaggle-drawing-with-llms



In summary, we allocated 36 seconds for SVG generation and aesthetic scores computation, and the remaining 27 seconds for candidate selection.

**Image Generation**
===========
We tried lots of open-sourced SD models (such as Sana, SD 3.5) etc and stuck with **SDXL-Flash** as it was the fastest model which could generate "nice-looking" images that met our expectations.

Using the default settings (1024x1024), SDXL-Flash took 8 seconds to generate an image. However, we improved the speed by up till **80%** by generating images with lower resolution. For 512x512, we were able to generate 18 candidate images within 30 seconds, without compromising much quality. It was validated locally that 512x512 or 640x640 gave the best tradeoff between the speed and accuracy. On the LB, 704x704 did pretty well too but we didn’t select it due to the low validation score.

Additionally, we used the prompt ```f'Cartoon style of "{description}".'``` as we noticed that the cover page of the DOODL paper showed an image that appeared to have a cartoon style after aesthetic optimization. When examining the SVG outputs, it was found that there is less quality loss in the conversion process, compared to a normal image.



**Bitmap to SVG**
==========
The bitmap_to_svg_layered technique shared publicly took around 4.5 seconds to run per image, and has significant information loss when compressed to 10k bytes.

When doing function profiling, we noticed that cv2's KMeans was the main culprit. Therefore, we sped it up using NVIDIA's cuML KMeans on GPU, with that, each image was converted to SVG in ~0.3 seconds (**15x speedup**).

Additionally to squeeze in more elements into 10k bytes, we embedded the following techniques. 
- Rounded all coordinates to the nearest integer
- Turned all ```<polygon>``` into ```<path>```
- Applied colour grouping (embed all paths of the same color within a ```<g>``` tag)
- Remove information bytes (in the OCR decoy function)
- Used [scour](https://pypi.org/project/scour/) to further trim the size of the final SVG

This allowed us to fit in 60 -> 168 elements (on average) for an improvement of **280%** without any quality loss compared to the publicly shared function

The OCR decoy by @richolson was used to mitigate hallucinations. We did not make any changes since a failure rate of 1/150 is deemed acceptable

**Candidate Selection**
==================
To quicken the aesthetic computation locally, we sped up apply_median_filter using cv2 and apply_fft_low_pass in ImageProcessor using NVIDIA CuPy (0.5 -> 0.3 s/image)

```py
class FasterImageProcessor(metric.ImageProcessor):

    def __init__(self, image: Image.Image, seed=None):
        super().__init__(image, seed)

    def apply_median_filter(self, size=9):
        """Fast median filter using OpenCV (≈10-20× PIL speed)."""
        # Pillow → NumPy (BGR order is okay; cv2 works on uint8 directly)
        img_array = np.asarray(self.image)
        # OpenCV expects odd kernel sizes ≥3
        if size % 2 == 0:
            size += 1
        filtered = cv2.medianBlur(img_array, ksize=size)
        self.image = Image.fromarray(filtered)
        return self

    def apply_fft_low_pass(self, cutoff_frequency=0.5, device=0):
        x = cp.asarray(self.image, dtype=cp.float32)      # H×W×3
        f = cp.fft.fftshift(cp.fft.fft2(x, axes=(0,1)), axes=(0,1))
    
        rows, cols = x.shape[:2]
        crow, ccol = rows // 2, cols // 2
        r = int(min(crow, ccol) * cutoff_frequency)
    
        y, xg = cp.ogrid[:rows, :cols]
        mask = ((y - crow) ** 2 + (xg - ccol) ** 2) <= r * r
        f *= mask[..., None]                              # broadcast over channels
    
        img_back = cp.fft.ifft2(cp.fft.ifftshift(f, axes=(0,1)), axes=(0,1)).real
        img_back = cp.clip(img_back, 0, 255).astype(cp.uint8).get()
        self.image = Image.fromarray(img_back)
        return self
```

We estimated the VQA score by asking one question: ``` 'Does <image> portray "{}"? Answer yes or no.' ```

Given that
| Method | Time Taken/image |
| --- | --- |
| Aesthetic Score | 0.3s |
| VQA + OCR Score| 3s |

Instead of estimating VQA of all images, we employed a reranker approach to estimate which images are more likely to get a higher score on competition metric, which allowed us to generate more candidate SVGs within the time limit. 

- We first generated N images (N depends on the image resolution), converted them into SVGs and computed the aesthetic score of all N images, with a time limit of 36 seconds.
- Since aesthetic alone doesn’t tell us how relevant the image is to the prompt, we used the [siglip](https://huggingface.co/google/siglip-so400m-patch14-384) model to compute the alignment score between the SVG and the prompt - multiplying them to get a rerank score. This improved the result significantly for 384x384 and 512x512, but not so much for 640x640
- The SVGs were ranked in descending order of the rerank score and VQA/OCR was attempted until the time exceeded 63 seconds (on average, it can run 7 such evaluations). 
- The SVG with the highest proxy fidelity score was chosen.

Notes
==========
- We tested our pipeline on our validation set of 500 examples (dataset to be shared later), and found it to be relatively stable on both CV and LB, with scores around 0.705-0.715 on public and 0.700-0.705 on private. The CV was correlating well up till 0.7 LB scores. 
- However, there was one submission that stood out (0.717 private, gold zone) and it was just a small change from our selected submission 😢 - the validation and public LB scores was not able to explain that (probably lucky OCR?). EDIT: it probably is a fluke, the resubmission got 0.709
- We did not try vtracer or any other methods to refine the SVG, probably would have done it if there was a few more days.
- We did not try to fine tune any stable diffusion model.

Failed Ideas
===============
- An additional deduplication step using similarity matching
- Training LLM to generate real-time questions during inference
- Adding backgrounds to images with the hope of improving aesthetic score
- Using more complicated reranking techniques without estimating VQA score. Our best public/private score without estimating VQA/OCR is 0.701/0.698 (with 25 candidate images at 512x512 resolution)
- Using 3b PaliGemma model to estimate VQA and OCR
