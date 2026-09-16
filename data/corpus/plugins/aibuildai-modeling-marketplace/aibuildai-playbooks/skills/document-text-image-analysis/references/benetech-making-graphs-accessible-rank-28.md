# 28th Place Solution

Competition: benetech-making-graphs-accessible
Rank: #28
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418705

First of all, we would like to thank Kaggle & Benetech for hosting this very interesting competition. It was great fun to participate, and we learned a tremendous amount from it. 😀

We used a combination of image classification, object detection, Donut, PaddleOCR, TrOCR, and lots of postprocessing! 😅

---

# Models

We trained models for the following tasks:

1. Plot type classification - ResNet34
2. Rotated vs non-rotated X tick label classification - ResNet34
3. Horizontal bar detection - FasterRCNN with ResNet50-FPN backbone
4. Vertical bar detection (used in the histogram pipeline as well) - FasterRCNN with ResNet50-FPN backbone
5. Line point detection - FasterRCNN with ResNet50-FPN backbone
6. Scatter point & dot detection (a single model for both) - FasterRCNN with ResNet50-FPN backbone
7. Tick mark detection - FasterRCNN with ResNet50-FPN backbone
8. X tick label detection (used only for detecting non-rotated X tick labels) - FasterRCNN with ResNet50-FPN backbone
9. Y tick label detection - FasterRCNN with ResNet50-FPN backbone
10. Donut (used only for recognizing rotated X tick labels) - <a href="https://huggingface.co/naver-clova-ix/donut-base" target="_blank">`'naver-clova-ix/donut-base'`</a>

In addition, we used the following models as-is (i.e., without any fine-tuning):

1. PaddleOCR for multi-line text detection & conversion to single-line texts - <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/PP-OCRv3_introduction_en.md" target="_blank">PP-OCRv3</a>
2. TrOCR for OCR of single-line texts - <a href="https://huggingface.co/microsoft/trocr-base-printed" target="_blank">`'microsoft/trocr-base-printed'`</a>

**Note:** We used PaddleOCR because TrOCR can only read single-line texts. For single-line texts, we found TrOCR to be very accurate.

---

# Inference Examples

The following images are examples of inference using each of the above models:

**Plot type classification:**



**Rotated vs non-rotated X tick label classification:**



**Horizontal bar detection:**



**Vertical bar detection (used in the histogram pipeline as well):**



**Line point detection:**



**Scatter point & dot detection (a single model for both):**





**Tick mark detection:**



**X tick label detection (used only for detecting non-rotated X tick labels):**



**Y tick label detection:**



**Donut (used only for recognizing rotated X tick labels):**



**Note:** We trained our Donut model only on image patches below the X axis which contain rotated text.

**PaddleOCR (for multi-line text detection & conversion to single-line texts):**



**TrOCR (for OCR of single-line texts):**

Image snippet:



Output: `"A. Total Genotoxic Potency"`

---

# Deep Learning Frameworks

We used:

- For image classification - <a href="https://docs.fast.ai/" target="_blank">fastai</a>
- For object detection - <a href="https://airctic.github.io/icevision/0.12.0/" target="_blank">IceVision</a> + fastai
- For Donut - Hugging Face Transformers <a href="https://huggingface.co/docs/transformers/main_classes/trainer" target="_blank">Trainer</a> API

---

# Data

In addition to the competition dataset, we used some additional images from the *ICPR 2022 CHART* dataset available at: https://chartinfo.github.io/toolsanddata.html

Bounding box annotations for the chart elements were added using either (i) custom Python scripts or (ii) manual annotation tools, viz. Make Sense (https://www.makesense.ai/) and Roboflow. The annotations were created in the PASCAL VOC format.

For line point detection, scatter point detection and tick mark detection, we had to convert the original point annotations to bounding box annotations. We noticed that the choice of bounding box size has a big impact on performance. Hence, we experimented with a few bounding box sizes, and found that the following sizes work well:

| Line Point | Scatter Point | Tick Mark |
|---|---|---|
| 18px | 16px | 16px |

For X tick label detection and Y tick label detection, we noticed that adding a 2px margin to the provided bounding boxes improves performance.

Finally, we found that in a tiny fraction of images, one or more bounding boxes overstep the bounds of the image. We deleted these images from our dataset.

---

# Model Validation Schemes

**Plot type classification:**



**Rotated vs non-rotated X tick label classification:**



**Horizontal bar detection:**



**Vertical bar detection (used in the histogram pipeline as well):**



**Line point detection:**



**Scatter point & dot detection (a single model for both):**



**Tick mark detection:**



**X tick label detection (used only for detecting non-rotated X tick labels):**



**Y tick label detection:**



**Donut (used only for recognizing rotated X tick labels):**



---

# Pipelines

## Image Classification



## Horizontal Bar Pipeline



## Vertical Bar Pipeline



## Histogram Pipeline



## Line Pipeline



## Scatter Pipeline



## Dot Pipeline



---

# Results

|   | Overall | Scatter | Dot | Line | Vertical Bar | Horizontal Bar |
|---|---|---|---|---|---|---|
| public | 0.66 | 0.06 | 0.00 | 0.26 | 0.33 | 0.01 |
| private | 0.49 | 0.19 | 0.01 | 0.08 | 0.20 | 0.01 |

---
