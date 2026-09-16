# 32nd Solution

Competition: czii-cryo-et-object-identification
Rank: #32
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561677

# **Acknowledgements**
We sincerely appreciate Kaggle and the competition organizers for offering this invaluable opportunity. We also extend our gratitude to @hengck23 , @fnands and @sjtuwangshuo for their significant contributions. Lastly, I would like to sincerely thank my teammates for their dedication and hard work during this time! @snnclsr , @miyamotodaiya and @yingpengchen 


# **Model**
We used the basic UNet3D model provided by MONAI as the primary model, and we implemented DLinkNet3D as an auxiliary model using Torch. Given the limited number of training samples, the model may face issues with generalization. To address this challenge, we designed a memory module to enhance the model's generalization ability. The specific design of the module is as follows:
[Mem Block]
This module was added to the bottom layer of the UNet model to enhance the generalization ability of high-dimensional vectors. 

The final models used are as follows:
|model name|nums|
|:--------:|:--------:|
|MemUNet|3|
|UNet|4|
|DLinkNet|1|

Parameter Settings:

        Channels：(48, 64, 80, 80)
        Strides：(2, 2, 1)
        num_res_units：2
        Dropout：0.3
        Mem Vector Size：(256,256)


# **Training**
We scaled the original data's radius to 0.48 or 0.5 (thanks to @miyamotodaiya for the experiment). After testing various training sizes (such as 96, 128, 136, 144, 164, and 176), we found that sizes 128 and 164 yielded the best results. Additionally, @yingpengchen tested different xyz size combinations, and the (48, 256, 256) size performed best. For the output channels, we tried 6, 7, and 8 output channels, with 6 channels providing the best performance.

## **Data Augmentation:**
We used the following data augmentation methods:
- RanRandCropByLabelClassesd
- RandFlipd
- RandRotated
- RandAffined
- RandGridDistortiond
- RandCoarseDropoutd
- RandScaleIntensityd
- RandShiftIntensityd
We set the probability of rotation and flipping to 1 to ensure data diversity.

## **Optimizer:**
We used the schedulefree.AdamWScheduleFree optimizer introduced by @miyamotodaiya .

## **Loss Function:**
- **Weighted Tversky Loss**
- **Distance Loss** : This loss function performs MSE loss after applying a distance transformation to the ground truth labels, making the model focus more on the central region of the labels. This loss function performs especially well when label overlap occurs.

## **EMA:**
We employed a method of dynamically adjusting the decay parameter based on the comparison between the current model's score and the best score. This approach yielded an improvement of approximately 0.001 in local tests.

# **Inference**
For inference, we used a sliding window strategy. Models trained at a size of 96 were used for inference at a size of 128 (since the DLinkNet model is too large to infer at a larger size), while other models were used for inference at sizes 176 or 180. The overlap was set to either 0.15 or 0.5.

## **Inference Time Optimization:**
We adopted two inference strategies:
1. Multi-Model Sliding Window Inference:
This approach combines multiple models with a sliding window for inference. The final score for this strategy was lb 763. We distributed the models across two GPUs, used multi-processing and TensorRT acceleration, and ran multiple models in parallel. With 7 models, the inference was completed in about 4 hours.

2. Fewer Models with Sliding Window and Extensive TTA:
This strategy used fewer models combined with a sliding window and extensive test-time augmentation (TTA), such as flipping and rotating. The final score for this strategy was lb 756. We loaded all models onto two GPUs, split the data into two parts, and used multi-processing and TensorRT acceleration to infer two datasets simultaneously. With 1 model and 7 TTA methods, the inference was completed in about 5 hours.

Finally, using the DataFrame fusion strategy provided by @miyamotodaiya , we merged the results from the two strategies, achieving a final score of lb 768.
[pipline]
# **Code**

Train Code: [https://github.com/wplll/Memory-Enhanced-3D-Segmentation](https://github.com/wplll/Memory-Enhanced-3D-Segmentation)

Multi-Model: [https://www.kaggle.com/code/peilwang/czii-infer-multi-model](https://www.kaggle.com/code/peilwang/czii-infer-multi-model)

Fewer Models and Extensive TTA: [https://www.kaggle.com/code/peilwang/czii-infer-fewer-models-and-extensive-tta](https://www.kaggle.com/code/peilwang/czii-infer-fewer-models-and-extensive-tta)

A big thank you to my teammates for their hard work and support!
