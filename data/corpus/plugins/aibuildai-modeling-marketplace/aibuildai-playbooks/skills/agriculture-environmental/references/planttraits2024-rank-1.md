# 1st Place Solution - PlantHydra 🐍

Competition: planttraits2024
Rank: #1
Source: https://www.kaggle.com/c/planttraits2024/discussion/510393

# PlantTraits2024 Solution Overview

Hello everyone!

Firstly, I'd like to extend my gratitude to the organizers and all participants for making this contest a fantastic experience. I'm thrilled to share that my solution secured 1st place in the PlantTraits2024 competition! Below, I outline my approach and key methods used. The full code and implementation details are available on my [GitHub repository](https://github.com/dysdsyd/PlantTraits2024). Feel free to reach out with any questions or for further discussion.

## Overview of My Solution:
<p align="center">
  [solution]
</p>
*Image Credits:  DALL-E*

My solution involved a combination of multiple methods coming together, which I'll summarize here. The primary components are:

### 1. The Three-Head Solution ("PlantHydra")

Initially, I used only one head to directly regress the traits, but this approach was insufficient for accurately identifying plant traits. There is a clear correlation between plant traits and species, and even species from similar environments share common traits. For example, species from regions with higher rainfall typically have greener and larger leaf areas. In the PlantTraits contest, we estimate mean trait values for a species, making it crucial to correctly identify the species or a cluster of species with similar traits.

With this in mind, I divided the training observations into 17,396 species based on unique plant traits, as explained in this [notebook by Rich Olson](https://www.kaggle.com/code/richolson/view-plants-by-species). I then adopted a three-head approach:

- **Regression Head:** Estimates the normalized traits directly.
- **Classification Head:** Classifies the observation into one of the 17,396 species, and the corresponding species traits are chosen as the predicted traits.
- **Soft Classification Head:** Addresses long-tail classification issues and inter-species correlation by calculating the final trait values through a weighted sum of species traits, with weights derived from the classification head's softmax scores.

Finally, these three heads were blended to obtain the final traits. The blending weights were made trainable to optimally weigh predictions from each head.


### 2. DINOv2 VIT-b and VIT-l Backbone

I experimented with ViT-b and ViT-l backbones for feature extraction due to their powerful performance in such tasks. A model pre-trained on the flora of southwestern Europe (based on Pl@ntNet collaborative images [1]) provided a significant boost in detecting plant traits.

### 3. Structured Self-Attention for Metadata

The metadata about climate, soil, etc., played a crucial role in estimating traits like nitrogen content and seed dry mass, as these factors directly affect plant growth. After several failed attempts with various consolidation approaches, such as PCA, I implemented a Structured Self-Attention module to identify correlations between traits and metadata, as well as within the metadata itself.

### 4. Objective Functions
This is the interesting part, as multiple objective functions were at play, and the weights for each were fine-tuned manually. Here is a breakdown by heads:

- **Regression Head:** R2 loss was used on normalized traits along with cosine similarity. It was essential not only to estimate individual traits accurately but also to maintain the correlation between all traits. Traits can be considered as a vector of dimension 6, and higher cosine similarity acted as a regularization to ensure the correct prediction of all traits.
- **Classification Head:** Focal loss was used to handle the long-tail classification task.
- **Soft Classification Head:** There was no dedicated loss for this head, as it was generated from the classification head itself.
- **Blending Heads:** A final R2 loss was used on unnormalized trait values, thereby allowing gradients to flow to all the heads.


### 5. Fine Tuning with Dedicated Schedulers
Fine-tuning the heads and backbone offered significant potential, as maintaining a delicate balance between underfitting and overfitting the model was crucial. Schedulers and optimizers played a key role in training the layers. Instead of using a single optimizer for both heads and the backbone, different learning rate schedules were employed.
[scheduler]
In summary, the heads with no prior knowledge were given higher learning rates and were warmed up earlier. In contrast, the backbone layers had scaled-down learning rates to prevent overwriting useful information.


### 6. Mixture of Experts

Finally, various model flavors were trained, some excelling in regressing traits directly and others in classifying species. The soft classification head outperformed individual models, striking a balance between hard classification and regression. The winning submission combined different heads from various models, weighted manually.

### Things That Didn't Work Out

- **Cut-Mix Augmentations:** Intended to improve generalization, but results were unsatisfactory.
- **SD Variation in Target Traits:** Although traits SD data was available, it was not useful for predicting mean traits of species in this competition. However, this data might be valuable for generalizing models on actual traits in future.

## Conclusion

In summary, the blend of regression, classification, and soft classification heads, combined with advanced feature extraction and fine-tuning strategies, led to the winning solution. Feel free to reach out for any questions, and the full code is available on [GitHub](https://github.com/dysdsyd/PlantTraits2024).

>### References:
[1] @misc{goeau_2024_10848263, author = {Goëau, Hervé and Lombardo, Jean-Chirstophe and Affouard, Antoine and Espitalier, Vincent and Bonnet, Pierre and Joly, Alexis}, title = {{PlantCLEF 2024 pretrained models on the flora of the south western Europe based on a subset of Pl@ntNet collaborative images and a ViT base patch 14 dinoV2}}, month = mar, year = 2024, publisher = {Zenodo}, doi = {10.5281/zenodo.10848263}, url = {https://doi.org/10.5281/zenodo.10848263} } https://zenodo.org/records/10848263
