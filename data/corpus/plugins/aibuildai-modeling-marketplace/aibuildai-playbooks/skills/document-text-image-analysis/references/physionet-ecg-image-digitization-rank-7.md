# 7th place solution

Competition: physionet-ecg-image-digitization
Rank: #7
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/7th-place-solution

Thanks to the PhysioNet hosts and Kaggle for this fun competition. There were many stages to optimize, and we took the opportunity to learn as much as we could from all stages. Congrats to everyone who competed, Harshit and I are excited to read through all the solution write-ups!

## TLDR

Our pipeline uses rotation, lead detection, lead segmentation, digitization, and out-of-distribution detection models. We modified the ECG-image-kit repository to create lead detection training data, relied on the competition data for digitization, and used out-of-distribution detection models to optimize our ensemble. For more details, keep reading!

## Cross Validation

For validating experiments, we used a k-fold cross-validation scheme across all samples. We used lightweight models on all folds to detect edge cases; for more computationally expensive models, we only validated on 100 samples. We found that 100 samples were enough for a strong CV/LB correlation and increased the speed of experiments.

## Data Generation

Next, due to a lack of lead annotations in the competition data, we modified ECG-image-kit to create data for the rotation, lead detection, and lead segmentation models. We updated the codebase to insert ECG plots into backgrounds, simulate shadows, and use more variable colors and textures in the ECG plots. All modifications we made to the codebase were done in an attempt to make the artifacts more realistic.

We used the [PTBXL dataset](https://physionet.org/content/ptb-xl/1.0.3/) for the raw ECG values, and the [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) for backgrounds and shadows. Here are a few samples of what we generated with bounding boxes and segmentation labels overlaid.

[IMG_0]

## Rotation

The first model in the pipeline was a simple classification model to predict when an image needed rotation. We used the B4 and B5 variants from the HGNet-V2 model family to predict 4 classes (0,90,180 or 270 degree rotation). We detected 69 images that required rotation in the training dataset, and applied this model first during inference.

## Lead Detection / Segmentation

Next, we trained a set of hybrid lead detection/segmentation models. This was a great learning curve for me (Bartley), as I have always wanted detection models that come without a confusing license. To do this, we designed a model to predict objectiveness, class scores, and offsets. It was important to add sufficient capacity in the bounding box detection head (>=128 channels) for the model to be able to learn the signal. We found that the ConvNeXt model family worked best, though the architecture supports any backbone from the timm library. 

We first ran the detection model to locate the AOI (area of interest). We then cropped the image and re-ran the model for a more precise result. We also added a minimum crop height of 16 pixels and a width of 64 pixels to account for small lead crops. This fixed all catastrophic detections we observed in the training set, and boosted LB by ~0.2-3. Here are some sample predictions on the training set.

[IMG_1]

We also added a segmentation branch to predict the pixels corresponding to the 13 different lead classes. We used this predicted segmentation as an input to our digitization models in the next stage.

## Digitization (Bartley)

The first digitization model we used was a `maxxvitv2_nano_rw_256.sw_in1k` model with a 1D unet decoder. We pooled the encoder features before passing them into the decoder. The model input was a 5-channel image. Three RGB channels, one for the target lead probability, and one for the maximum probability of other leads. The 5-channel input helped generate more robust predictions on crops with overlapping leads.

[IMG_2]

During training, we applied heavy color, distortion, rotation, horizontal flips, vertical flips, shifts, and thin coarse dropout (to simulate pen marks). We also implemented a custom Albumentations module to add ECG-related keywords/phrases to each image, though it was unclear how much this augmentation improved the models. We were unable to converge this model completely, and were still seeing gains at the end of the competition. We believe that more computing power could lead to further performance improvements with this architecture.

We used a couple of variations of SNRloss. During the low-resolution stage, we used a variation of SNRloss that forces the model to learn the optimal vertical shift. In the later stages (once vertical shift was learned), we used a variation that used the optimal vertical shift to more closely align with the competition metric. The latter approach was able to achieve higher final scores.

For the lead II full crops, we used a VIT model with a linear head to go from patch embeddings to pixel-level predictions. This architecture was identical to Harshit's, and we will go into more details in the next section.

## Digitization (Harshit)

For our next set of digitization models, we rely on the same preprocessing steps from Bartley's pipeline. All our models here used a `vit_small_patch16_dinov3.lvd1689m` encoder with slight differences in the training setup. The architecture was heavily inspired by Harshit’s 1st Place Solution in the Yale Competition [here](https://www.kaggle.com/competitions/waveform-inversion/writeups/harshit-sheoran-1st-place-solution).

[IMG_4]

We developed three variations of this model to improve diversity. While the architecture and training method remained consistent, we varied the input data sources (crops) and loss functions. We used random padding augmentation during training on crops derived from the sources below.

| Model Version | Input Data Sources (Crops) | Loss Function |
| :--- | :--- | :--- |
| **V27** | `train_crops5`, `train_bartley_crops3` | MAE |
| **V27-SNR** | `train_crops5`, `train_bartley_crops3` | SNR |
| **V6** | `train_gen_crops1`, `train_crops4`, `train_bartley_crops2`, `train_bartley_crops4` | MAE |

In addition, we employed a multi-stage training approach with progressive image size scaling to ensure stable convergence. Each stage consisted of **20 epochs**, scaling up the resolution as follows:

    `224x896` → `224x1782` → `224x2688` → `224x3584` → `336x3584`

We intentionally kept augmentations minimal, only using horizontal flips during training. Despite the light augmentation pipeline, the models converged effectively. To test the robustness of this pipeline, we took pictures of ECG plots on different monitors to simulate a distribution shift. Scores were consistent with those on the competition set indicating that the models were robust.

## Ensemble

We expected a large boost when combining our approaches as we used a diverse set of architectures, training pipelines, and loss functions. Harshit’s models excelled on in-distribution samples and when there was a significant drift in the signal. Bartley’s models excelled on out-of-distribution samples and on crops with overlapping leads. A simple mean ensemble of our pipeline scored **22.54/22.10** on the Public/Private LB.

### Out-of-Distribution (OOD) Detection

To further improve ensemble performance, we implemented an out-of-distribution (OOD) detection model. Since Harshit’s models were highly specialized for in-distribution samples, we wanted to detect when to mask his predictions during inference.

To do this, we trained a feature extractor using `tf_efficientnetv2_s` and ArcFace Loss. During inference, we calculated the embedding of each test image and compared it to the average embedding of each image type in the training set. If the Cosine Similarity between the test image and any of the average embeddings was <0.5, we masked Harshit's predictions. This strategy boosted our score further to **22.80/22.48**.

All the models we trained in this competition (Harshit and Bartley) used the Muon optimizer from timm. We found that this significantly outperformed all others and thought it was worth a mention.

## Final Note

Last thing, a quick shout-out to @TheoViel. I recently modified my training pipeline to follow a similar structure to his [RSNA 2023 Solution](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma). It’s an excellent repository that I would recommend checking out.

Thanks for reading, and as always, Happy Kaggling!

Code: [here](https://github.com/brendanartley/PhysioNet-Competition)

Datasets: [here](https://www.kaggle.com/datasets/brendanartley/physionet-2025-submission), [here](https://www.kaggle.com/datasets/brendanartley/physionet-2025-submission---other-data)

Inference: [here](https://www.kaggle.com/code/harshitsheoran/physionet-infer-v-final)
