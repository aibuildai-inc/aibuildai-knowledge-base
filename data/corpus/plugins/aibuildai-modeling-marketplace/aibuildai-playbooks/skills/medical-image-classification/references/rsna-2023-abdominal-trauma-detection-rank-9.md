# 9th place solution

Competition: rsna-2023-abdominal-trauma-detection
Rank: #9
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447506

First of all, I would like to express my gratitude to the Kaggle staff for hosting this fantastic competition, as well as to the organizers at RSNA. I also want to extend my appreciation to all the hardworking participants who dedicated themselves to this competition. Special thanks go to my teammate @liushuzhi for his tireless efforts and insightful discussions.


# **Overview**



# **Details of Each Model**

### **3D Segmentation**

- Used the implementation from @haqishen's top solution of the previous competition.  [[1](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607)] (Thank you!)
- Input size: (128, 128, 128)
- Models used: ResNet18d, UNet

### Classification - **Liver/Kidney/Spleen**



- Developed a single model to classify all three organs instead of individual models for each organ due to better performance.
- For patients with 2 series, only used the one with the lower aortic_hu for training. For patients with 1 series, used all of them. This decision was based on experimentation rather than a specific rationale.
- Performed cuboid crop using the 3D segmentation masks, slightly expanding the cuboid to preserve edges.
- Weighted the loss during training according to the competition metrics [1, 2, 4].
- Model architecture: SE-ResNeXt → LSTM → Concatenation of [1D CNN, Attention]
- Input size: (96, 4, 256, 256)
    - Channels are (i-1, i, i+1, mask), with 'i' representing the index of images along the z-axis.

### Classification - **Bowel**

Two-stage training was employed to leverage image-level labels.



**1st Stage**

- Binary classification with a 2D model using image_level_label.csv.
- Similar to liver/kidney/spleen model, cuboid crop was done using 3D segmentation masks to create 4-channel input data. Thus, input size was (4, 384, 384).
- Sampled data randomly to achieve a 1:10 ratio between positive and negative.
- The loss is weighted based on the competition's metrics.
- SE-ResNeXt was used for the model.

**2nd Stage**

- Aggregated features obtained from the model trained in the 1st stage.
- Model architecture: SE-ResNeXt → LSTM
- Weights from the 1st stage were loaded and kept frozen, updating only the network beyond the LSTM during training.

### Classification - **Extravasation**

Similar to the approach for Bowel, a two-stage training strategy was used to leverage image-level labels. The model and aggregation methods were the same, but there were some differences.

- Cuboid crop was not performed.
- Since anomalies were small, its resolution was increased to (96, 3, 512, 512).
- Used data with higher aortic_hu for patients with 2 series to highlight features.


# **Post-Processing**

A post-processing step was introduced to improve the optimization of any_injury. The only post-processing step we performed was simply multiplying the coefficients.

Coefficients were searched and applied based on OOF prediction values. Although the original plan was to create a stacking model for this purpose, it was simplified due to time constraints.

```python
low_coef = 1.5
high_coef = 1.5
ev_coef = 1.5
bowel_coef = 1.0

df_pred["liver_low"] *= low_coef
df_pred["liver_high"] *= high_coef
df_pred["kidney_low"] *= low_coef
df_pred["kidney_high"] *= high_coef
df_pred["spleen_low"] *= low_coef
df_pred["spleen_high"] *= high_coef
df_pred["extravasation_injury"] *= ev_coef
df_pred["bowel_injury"] *= bowel_coef
```

# **CV**

| | bowel | ev | kidney | liver | spleen | any_injury | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w/o Post-Processing | 0.1293 | 0.5348 | 0.3146 | 0.4192 | 0.4454 | 0.5533 | 0.3994 |
| w/ Post-Processing | 0.1293 | 0.5303 | 0.3141 | 0.4190 | 0.4485 | 0.4925 | 0.3889 |

# **Source Code**
|  | URL |
| --- | --- |
| Inference notebook | https://www.kaggle.com/code/kapenon/rsna2023atd-9th-place-inference | 
| Training code | https://github.com/kapenon/rsna2023atd_9th_solution | 

# Acknowledgments
We want to extend our thanks to the Kaggle staff and hosts who organized this fantastic competition, all the dedicated participants, and Rist Inc. for their support in providing computational resources.
