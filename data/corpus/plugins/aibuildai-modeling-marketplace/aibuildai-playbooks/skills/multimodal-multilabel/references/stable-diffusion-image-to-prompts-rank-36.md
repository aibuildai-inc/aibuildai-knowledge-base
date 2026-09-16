# 36th place solution

Competition: stable-diffusion-image-to-prompts
Rank: #36
Source: https://www.kaggle.com/c/stable-diffusion-image-to-prompts/discussion/410609

No shakeup. Congrats to all the winners!!

This is my **second solo silver**. And I'm finally becoming a **Kaggle competition Master** after the result finalises!!!
***
My final ensemble consists of fine-tuned models and zero-shot models.

**TLDR:** KNN regression + CLIP interrogator + an ensemble of 4 fine-tuned models, with manual weights of 0.3, 0.1 and 0.6 respectively. The submission notebook can be found [here](https://www.kaggle.com/code/xfffrank/sd-clip-knnregression-vit-interrogator/notebook).

The details are explained as below.

## Fine-tuned models

### Data preprocessing

For this part, or even the whole competition, **data preprocessing and the diversity of datasets** are two of the most important points. For the training of each single model, I used texts from Diffusion-2M, Diffusion-14M, the public 900k and 80k datasets.

There are two reasons why preprocessing is necessary:

1. Filter high-correlation data so that the model is not easy to overfit.
2. Reduce the size of training set so that the training cost is acceptable.

For each dataset, the preprocessing follows this pipeline:

1. Remove duplicates according to existing dataset.
2. Filter by hand-crafted rules.
    1. Remove texts with num_of_words < 5.
    2. Remove duplicates for texts that have the same 4 starting/ending words.
    3. Remove texts that have non-English characters.
    4. Remove texts that have more than 77 tokens, using the same CLIPTokenizer as Stable Diffusion v2.
3. Filter by correlation(i.e. cosine similarity) within the dataset. I used a threshold of 0.8.
4. Filter by correlation with existing datasets. I used a threshold of 0.8.

> When filtering by correlation, it's important to utilize GPU power and process the dataset in batches. For instance, when utilizing the `encode` function provided by the "Sentence Transformer" library, enabling the "normalize_embeddings=True" parameter will return embeddings with a unit length. This allows for easy computation of cosine similarity using `torch.matmul`.

### Data generation

I re-generated the images using Stable Diffusion v2 following the settings in the dataset description.

### Training

- CV split: Split the validation set according to the token length. Split into 12 ~ 15 folds and take the first fold as the validation set.
- How many layers to freeze during fine-tuning?
    - I found that the validation score did not increase after the number of training parameters reached a certain point, so I manually tune the starting unfreezing layer to make the number of training parameters as ~80 million.
- Data augmentation: the only two methods I found helpful are `HorizontalFlip` and `MixUp`.
- Number of epochs: 4. The validation score always reached the highest point at epoch 3.

### Ensemble

| Model | public LB | weight |
| --- | --- | --- |
| clip-vit-large-patch14-336 | 0.57954 | 0.3 |
| clip-vit-large-patch14 | 0.57665 | 0.2 |
| blip-image-captioning-large | 0.57621 | 0.2 |
| convnext_large_mlp.clip_laion2b_ft_soup_320 | 0.57907 | 0.3 |
- ensemble public LB: 0.59314
- ensemble public LB (normalise the outputs before performing ensemble): 0.59673

## KNN regression

Follows the [public notebook]([https://www.kaggle.com/code/motono0223/sdip-clip-knnregression-zeroshot-method](https://www.kaggle.com/code/motono0223/sdip-clip-knnregression-zeroshot-method)), except that I also added datasets processed on my own.

## CLIP interrogator

Follows the [public notebook]([https://www.kaggle.com/code/leonidkulyk/lb-0-45836-blip-clip-clip-interrogator](https://www.kaggle.com/code/leonidkulyk/lb-0-45836-blip-clip-clip-interrogator)).

## Final thoughts

There are several things I didn’t have time to explore:

1. Generate more datasets to increase the diversity.
2. Filter more public datasets and add them to the KNN model.
3. Explore more zero-shot methods.

I’m still new to the multi-modal field, but this competition is a good starting point!
