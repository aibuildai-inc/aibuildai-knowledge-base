# 13th Place Solution - God Bless CV

Competition: isic-2024-challenge
Rank: #13
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532654

First of all, thanks to Kaggle and the organizers for this interesting challenge. It brings back memories of the 2020 melanoma competition, where I won my first solo gold medal. That experience still holds a special place in my heart.

# **Solution Overview**

This solution combines multiple tree-based models with custom feature engineering, sampling techniques and multiple image based features.

# **The Image Baseline Part**

I began with a simple approach to measure the base capabilities of image models on skin images. For this purpose, I experimented with several models by extracting their feature maps and feeding them into basic linear and GBDT models without further fine-tuning of the backbones. This approach provided me with intuition about which models I should focus on for fine-tuning and further development.

In these experiments, I collected CV scores for several models, which averaged around 0.15. The best-performing model I found was eva_large_patch14_196, so I began my initial image fine-tuning efforts with this model.

# **Vision Model Finetunes**

In this part, I experimented with numerous variables based on my intuitions. Let's start with image transformations: Each image model required different input image sizes, so I had to convert them on the fly. However, I believed that changing the shape without maintaining the image aspect ratio might have detrimental effects, since we are dealing with skin lesions where their shapes could be highly significant on the diagnosis. Therefore, in my resizing process, I maintained the same aspect ratio and padded the rest of the image if needed.

The second point was dealing with class imbalance. I noticed that vision transformer models performed better with small batch sizes, but when using the full dataset, I was updating gradients with batches consisting entirely of class 0 samples. To address this, I decided to downsample the majority class and upsample the minority class (within the training fold, of course). However, this method introduces duplicate positive class samples in the same batches. To mitigate this issue, I applied heavy augmentations, ensuring that even with duplicates in the same batch, I had different versions of each image.

To take image augmentations one step further, I implemented MixUp augmentation in the training loop. This approach made the model more cautious when producing logits at the final stage.

The final list of models I chose for the next step includes:

1. eva_large_patch14_196
2. caformer_b36
3. swinv2_base_window12to16_192to256
4. nf_regnet_b1
5. edgenext_base

Each of these models individually achieved an OOF CV score above 0.155, with eva_large being the most successful, scoring 0.164. To stay within Kaggle's inference time limits, I implemented parallel inference using two T4 GPUs in half precision with a batch size of 512.

# **Ensemble Part**

For this part, I didn't deviate much from public notebooks, making only small adjustments. Maybe using the same validation folds from the image training was a crucial point. Essentially, I created a voting classifier based on LightGBM, XGBoost, and CatBoost, following a similar approach to @greysky's excellent tabular notebook.

# **One Last Trick**

I didn't invest a lot of time in creating tabular features, instead using what was publicly available. However, upon analyzing these features, I noticed a high number of useless correlated, low variance, or noisy features. Consequently, I wanted to quantify the impact of these features on the final score.

Using feature importances from individual GBDT models (based on tree-based gains or splits) didn't seem logical to me, as they tend to overfit and don't carry much significance in the final ensemble with a soft voting classifier.

Instead, I implemented a permutation feature importance scheme. For those unfamiliar with this method, it involves shuffling each feature n times and checking how it affects the final fold score on average. This approach allowed me to directly quantify the effect of a feature in the voting classifier setting without relying on individual models.

Based on this permutation test, I dropped a significant number of features. In total, I removed approximately 190 features from the tabular data.

This feature reduction led to substantial improvements in the CV score, increasing it from 0.179 to 0.183. This improved model is currently my best-performing private model, achieving a gold medal status!

# **Key Takeaways for Me**:

- Started with a simple image baseline to get a feel for different models' capabilities on skin images.

- Fine-tuned vision models with some tricks: keeping aspect ratios, balancing classes, and throwing in heavy augmentations including MixUp.

- Picked a handful of top-performing models, with eva_large being the star pupil scoring 0.164 on OOF CV.

- For the ensemble, didn't reinvent the wheel but tweaked public notebooks a bit, mainly keeping validation folds consistent.

- Realized the tabular features were a bit of a mess, so used permutation feature importance to separate the wheat from the chaff.

- Dropped about 190 features that weren't pulling their weight, which surprisingly boosted the CV score from 0.179 to 0.183 - hello, gold medal!
