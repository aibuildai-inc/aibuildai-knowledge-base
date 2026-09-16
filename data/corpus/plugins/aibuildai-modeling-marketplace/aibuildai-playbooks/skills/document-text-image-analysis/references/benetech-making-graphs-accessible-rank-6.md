# 6th place solution - deplot & UNet postprocessing

Competition: benetech-making-graphs-accessible
Rank: #6
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418466

Thanks to kaggle and benetech for hosting an interesting problem. It was a great team effort with @darraghdog 

## TLDR
Our solution is a 4 seed blend of a single deplot model for all chart types. For scatter plots we run additional post processing by using the number of scatter points predicted by a U-Net point segmentation model. 

## Data preprocessing

### Benetech dataset

For training we used all of benetech extracted data, and approx 20% of the generated data which was selected based on the hardest samples. This helped speed up the training pipeline significantly. 
Numerical values had a lot of digits in ground truth which did not match well with our loss function (cross entropy) in pix2struct based models. Therefore, it was important to round these numerical values. For this, we bucketed the labels to approx 5 buckets within the tick label axis. So, for example, the buckets in the below would be `0, 40000, 80000, 120000, 160000, 200000, 240000… etc.` and the y-axis label to learn would be `840000, 960000, 760000, 700000, 580000, 460000, etc.`. For ground truth labels which had no tick labels, similar rounding was performed based on the min-max range of the data series, using approximately 40 buckets. 



Within the pix2struct model, the labels were learned as key-value sequences, similar to the training of deplot. For the example above, that would be
 `<pad>Malaysia;960000|Malawi;840000|Maldives;760000|Mali;700000|Mauritania;660000|Melanesia;580000|Malta;580000|Mauritius;540000|Martinique;460000|Mayotte;420000<\s>`

A separate decoder was used to learn the reversed key-value sequence. This was linked to the same encoder and added as an auxiliary loss so the model would not focus more on either side of the image. Since the heavy part of deplot is the encoder, a double decoder did not create much overhead.

### External data 

For this competition, we utilized a commercially available subset of PubMed articles to train our models. Approximately 80% of the labeled samples in the ICDAR dataset were found to originate from this commercial dataset. We ensured that the use of this dataset complied with the applicable licensing terms. To expand our training data, we applied pseudo-labeling techniques followed by manual review. Specifically, we pseudo-labeled and manually reviewed around 8,000 bar and line graph samples from PubMed articles. The pseudo-labeling results were quite accurate on bars, requiring only verification that the identified charts were the correct type (e.g. removing box plots). Lines needed some corrections. We also used our U-Net model to pseudo-label approximately 1,000 scatter plot samples from PubMed, and then manually added any missing data points. Using the VGG Image Annotator tool, this manual annotation process was surprisingly fast and efficient. 

### Data generation
We used the great [public kernel](https://www.kaggle.com/code/brendanartley/benetech-5-chart-types-generator) by @brendanartley in two ways

We pretrained our deplot models using the 500k images downloadable the public kernel, which gave a small improvement, compared to using the original checkpoint from huggingface. 

We also tweaked the kernel a bit to generate more sophisticated scatter plots (more points, different markers, more overlap etc) and output x/y coordinates of each scatter point. We generated 50k scatter plots with that script and used them additionally for training the segmentation model

## Models

Deplot was used for predicting bars, lines and scatter; for scatter, we additionally used a u-net model. 

Deplot was trained for 20 epochs, with a learning rate of approx 5e-5 and batchsize 8. [Pixeldropout augmentation](https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py#L2413) was found to prevent overfitting. No other image augmentations were used. For experimentation, 1024 patches were used which decreased training time 2X with only a slight degradation in scores. The loss was a combination of cross entropy on the key value sequences mentioned above and the chart type. A linear layer on top of the first token of deplot was used to predict the chart type as an auxiliary.

For scatter plots we used a U-Net (EfficientNet-B7 encoder) + non maximum suppression (NMS) to predict number of scatter points and performed a simple yet effective postprocessing where deplot predicted a different number of points:

- If deplot predicted too many points, disregard the last ones
- If deplot predicted too few points, fill missing with the mean value

Below you can see the segmentation mask for U-Net training. Using the scatter point annotation we created “smooth” points with 10 pixel gaussian radius and drew onto an empty image. 



## Ensembling

For deplot we did inner ensembling by averaging decoder prediction in each greedy decoding step. For Unet we simply averaged predicted masks before performing NMS.

## Ablation study (Public LB Scores)

The following table gives a quick overview, how much each component contributed to our solution

| component | score |
| --- | --- |
| Baseline deplot@1024 tokens | 0.71 |
| using fullfit | +0.01 |
| 1024 -> 2048 tokens | +0.03 |
| Scatter pp Unet@512x512 | +0.05 |
| Pixeldropout | +0.01 |
| External data | +0.02 |
| Pretrain on 500k | +0.01 |
| Unet@512x512 -> 768x768 | +0.01 |
| 3x Unet@768x768 | +0.01 |
| 4x seed blend of deplots | +0.01 |
 **final score** | **0.87** 

## Used tools/ repos

- Unet:[ Segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch) + [timm encoder](https://github.com/huggingface/pytorch-image-models/tree/main/timm)
- Deplot: [huggingface](https://huggingface.co/google/deplot)
- logging/ visualization: [neptune.ai](https://neptune.ai/)
- augmentation: [albumentations](https://albumentations.ai/)
- labelling: [VGG Image Annotator tool](https://www.robots.ox.ac.uk/~vgg/software/via/)

Thank you for reading, questions welcome

edit:

June 21st: inference kernel made public: https://www.kaggle.com/code/christofhenkel/benetech-6th-place-dd/notebook
