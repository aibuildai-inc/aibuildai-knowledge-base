# 37th place solution - T0m part [Segment Copy-Paste & Progressive Learning]

Competition: tensorflow-great-barrier-reef
Rank: #34
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307669

Thanks to the hosts for providing an interesting competition, and congratulation to all winners.
This is my first detection competition and it was good experience.

I want to share my work's key points.

# Segment Copy-Paste Augmentation

We thought it was important for the model to learn to detect a small cots. So we tried augmenting the data by attaching a small starfish. However, we thought that a simple copy-paste would result in an unnatural image and overfitting of the background. Therefore, we ( @aerdem4 ) first train a segmentation model, created a segmentation mask, and then applied copy-paste augmentation using them.

[[2022-02-15-14-53-57.png]](https://postimg.cc/MXrry1SC)

As it is, the style is different, we transform style by shifting RGB value to meet the mean.
ex) segment[:, :, 0] = segment[:, :, 0] - (segment[:, :, 0].mean() - background[:, :, 0].mean())


# Progressive Learning

To create a model that is robust to noise and size, image size and augmentation were made larger and stronger as the epoch progresses (Progressive Learning)

[[2022-02-15-14-54-11.png]](https://postimg.cc/0bRXWPzn)


# Model
| Model | image_size | description | PublicLB | PrivateLB |
| --- | --- | --- | --- | --- |
| yolov5m | 3200 | CustomCopyPaste & Progressive Learning Epoch=25 | 0.629 | 0.699 |
| yolov5m | 3000 | Epoch=8 | 0.662 | 0.691 |
| yolov5x6 | 1920 | Epoch=12 | ??? | ??? |

I used TTA and Tracking.

train: video-0, 1
valid: video-2 (CV: 0.71~0.73)

# Ensemble
Ensemble by using wbf.
and, weighted conf for each detected bbox size.

ex)
if (bbox from model_1 and bbox_area < 1000):
    conf *= 3
elif (bbox from model_1 and bbox_area < 4000):
    conf *= 0.5
...
and then wbf

| Description | PublicLB | PrivateLB | sub |
| --- | --- | --- | --- |
| 3 Model | 0.638 | 0.711 | not selected |
| 3 Model re-train full-data| 0.638 | 0.696 | selected |

CV: 0.77~0.78

Thanks :)
