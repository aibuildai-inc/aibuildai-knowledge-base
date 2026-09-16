# Dinov3 + DoRA + Tweedie Loss + Ballanced R2

Competition: csiro-biomass
Rank: #31
Source: https://www.kaggle.com/c/csiro-biomass/writeups/dinov3-dora-tweedie-loss-ballanced-r2

# Introduction
- In this writeup, I will focus on some key points in this method. The baseline model is inspired by https://www.kaggle.com/code/mattiaangeli/dinov3-baseline-lb-0-70

# Preprocessing
- Use StratifiedGroupKFold to divide into 5 corresponding folds based on **labels="State**" and **groups="Sampling_Date"**. However, brute-force analysis can be performed on multiple random_seed to find the **golden_seed**. 
- The metric used to find the golden_seed is the Mean calculated based on the **"Dry_Total"** column of the training and validation sets.

# Model fine-tuning
- Use DoRA instead of LoRA. In short, LoRA (Low-Rank Adaptation) updates a model by adding a low-rank matrix to the original weights, focusing primarily on efficient fine-tuning. DoRA (Weight-Decomposed Low-Rank Adaptation) improves this by decomposing the weights into two components—magnitude and direction and applying LoRA only to the directional part. This allows DoRA to mimic full fine-tuning more accurately by learning changes in scale and orientation separately.
- Objective: Use Tweedie instead of L1 or L2 or Hubert Loss. You can look at the distribution of the original data through the image below:

It can be concluded that: 
- Zero-inflation: A significant number of samples have exactly 0g biomass. 
- Right-skewness: Non-zero values ​​are highly skewed with a long tail.

Therefore, using the Tweedie loss function is most suitable to describe this distribution. It is a compound Poisson-Gamma distribution. You can see more about the Tweedie distribution function here: https://en.wikipedia.org/wiki/Tweedie_distribution

# Ballance metric
One of the mistake is relying too much on the overall **R2** score provided by the organizers. A high overall **R2** can be misleading and may hide significant bias in your model.
Since we need a model that generalizes well across all three targets **"Dry_Green"**, **"Dry_Clover"**, and **"Dry_Dead"**. We should calculate the **R2** for each category separately. To prevent the model from performing well on one category while failing on others, I have introduced a custom evaluation metric:
$$S = \{(R^2_{\text{Green}}, R^2_{\text{Clover}}, R^2_{\text{Dead}})\}$$$$\text{Score} = \text{mean}(S) - \alpha \times \text{std}(S)$$
Why this works:
By subtracting the standard deviation (**std**) , we penalize inconsistency. This forces the model to achieve a balance between all categories, ensuring the predictions are stable and unbiased across the entire dataset.

The complete source code is here. Hopefully, it will be useful to everyone:
https://github.com/sunny442k3/CSIRO-Image2Biomass
