# 7th place solution

Competition: UBC-OCEAN
Rank: #7
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465697

Thanks to kaggle and UBC for hosting this interesting competition and congrats to all the winners for their hard work! I would also like to thank my teammates and everyone in the discussion forum for their help!

# Method

## Summary

Our final solution is based on multiple instance learning(MIL) for **ovarian cancer subtype classification** and use `sigmoid` and thresholding for **outlier detection**.
We did not use mask annotation and additional datasets in the final submission.

### 1. preprocess



1\. Use `pyvips` to speed up png reading speed. (Thanks for [GUNES EVITAN's pyvips notebook](https://www.kaggle.com/code/gunesevitan/libvips-pyvips-installation-and-getting-started).)

```python
image = pyvips.Image.new_from_file(image_id, access='sequential').numpy()
is_tma = image.shape[0] <= 5000 and image.shape[1] <= 5000
```

2\. Downsample WSI and TMA from x20 and x40 to x10 respectively. (Maybe x20 results will be better, but I can't submit due to resource constraints.)

```python
if is_tma:
    resize = A.Resize(image.shape[0] // 4, image.shape[1] // 4)
else:
    resize = A.Resize(image.shape[0] // 2, image.shape[1] // 2)
image = resize(image=image)['image']
```

3\. Deduplicate the identical tissue areas for WSI. (I'm not sure if this contributed to the results, but it saved me a lot of local memory.)

```python
def rgb2gray(image: np.ndarray):
    image = image.astype(np.float16)
    image = (image[..., 0] * 299 + image[..., 1] * 587 + image[..., 2] * 114) / 1000
    return image.astype(np.uint8)

if not is_tma:
    resize = A.Resize(image.shape[0] // 16, image.shape[1] // 16)
    thumbnail = resize(image=image)['image'].astype(np.float16)
    mask = rgb2gray(thumbnail) > 0
    x0, y0, x1, y1 = get_biggest_component_box(mask)

    scale_h = image.shape[0] / thumbnail.shape[0]
    scale_w = image.shape[1] / thumbnail.shape[1]

    x0 = max(0, math.floor(x0 * scale_w))
    y0 = max(0, math.floor(y0 * scale_h))
    x1 = min(image.shape[1] - 1, math.ceil(x1 * scale_w))
    y1 = min(image.shape[0] - 1, math.ceil(y1 * scale_h))
    image = image[y0: y1 + 1, x0: x1 + 1]
```

4\. Use the non-overlapping sliding window method to tile the tissue area into 256x256 patches. (For TMA I used overlap, but not sure if that would have an impact on the results.)

```python
def image2patches(image: np.ndarray, patch_size: int, step: int, ratio: float, transform, is_tma: bool):
    patches = []
    for i in range(0, image.shape[0], step):
        for j in range(0, image.shape[1], step):
            patch = image[i: i + patch_size, j: j + patch_size, :]
            if patch.shape != (patch_size, patch_size, 3):
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)))

            if is_tma:
                patch = transform(image=patch)['image']
                patches.append(patch)
            else:
                patch_gray = rgb2gray(patch)  # (patch_size, patch_size)
                patch_binary = (patch_gray <= 220) & (patch_gray > 0)

                if np.count_nonzero(patch_binary) / patch_binary.size >= ratio:
                    patch = transform(image=patch)['image']
                    patches.append(patch)

    if len(patches) != 0:
        patches = torch.stack(patches, dim=0)
    else:
        patches = torch.zeros(0, dtype=torch.uint8)

    return patches

image2patches(image, 256, [256, 128][is_tma], 0.25, transform, is_tma)
```

### 2. Subtype classification



Cancer subtype classification method is mainly based on multiple instance learning(MIL).
After trying various backbone and MIL methods, `CTransPath` and `LunitDINO` were finally selected as the backbone, `DSMIL` and `Perceiver` were selected as the MIL classifier. For their specific information, please refer to:

1. [CTransPath, MIA2022](https://github.com/Xiyue-Wang/TransPath)
2. [LunitDINO, CVPR2023](https://github.com/lunit-io/benchmark-ssl-pathology)
3. [DSMIL, CVPR2021](https://github.com/binli123/dsmil-wsi)
4. [Perceiver, BMVA2023](https://github.com/cgtuebingen/DualQueryMIL)

Local CV results:

| exp | CC | EC | HGSC | LGSC | MC | mean |
| --- | --- | --- | --- | --- | --- | --- |
| CTransPath + DSMIL | 0.9300 | 0.7657 | 0.8909 | 0.7822 | 0.7911 | 0.8320 |
| CTransPath + Perceiver | 0.9695 | 0.8147 | 0.8818 | 0.8044 | 0.9156 | 0.8772 |
| LunitDINO + DSMIL | 0.9400 | 0.7240 | 0.8864 | 0.8244 | 0.9356 | 0.8621 |
| LunitDINO + Perceiver | 0.9300 | 0.7983 | 0.8591 | 0.8711 | 0.8933 | 0.8704 |

Leaderboard results:

| exp | public | private |
| --- | --- | --- |
| CTransPath + LunitDINO + DSMIL | 0.57 | 0.54 |
| CTransPath + LunitDINO + Perceiver | 0.58 | 0.57 |
| CTransPath + LunitDINO + DSMIL + Perceiver | 0.6 | 0.58 |

I almost didn't adjust the MIL hyperparameters because I found that high CV score tended to be low public score.

1. For `DSMIL`, we use `nn.CrossEntropyLoss` as loss function.
2. For `Perceiver`, we use `nn.BCEWithLogitsLoss` as loss function and use `mixup`, `label smoothing` to alleviate overfitting.

### 3. Outlier detection

We tried many methods, two of which can get a private score of 0.6. (Private score 0.58 if not use outlier detection.)

#### 1. BCE + Thresholding

Score: public 0.6 and private 0.6.

This method is very simple. Use `nn.BCEWithLogitsLoss` as the loss function to train the model, and then for the maximum prediction probability, if it is less than 0.4, it is considered an outlier.

```python
logits = self.model(x)
probs = F.sigmoid(logits)  # (C,)
pred = probs.argmax(dim=0).item()
if max(probs) < PROB_THRESH:  # Choose the threshold based on the validation set
    pred = 5  # 'Other' class
```

#### 2. Probability entropy

Score: public 0.54 and private 0.6.

This method is also very simple. Compared to setting a probability threshold, this method detects outliers by calculating the entropy of the probability.

```python
logits = self.model(x)
probs = F.sigmoid(logits)  # (C,)
pred = probs.argmax(dim=0).item()
entropy = (probs * torch.log2(probs)).mean(dim=0)
if entropy > ENTROPY_THRESH:   # Choose the threshold based on the validation set 
    pred = 5  # 'Other' class
```

# Summary

## which didn't work

1. Extra dataset: ATEC, PTRC-HGSOC, CPTAC-OV, TCGA-OV, Bevacizumab.
2. End-to-end finetune the backbone and MIL together by selecting cancer areas through attention or mask.
3. Select only patches in cancer areas for MIL.
4. Detect outliers based on patch prediction probability entropy. ([MIA2023](https://www.sciencedirect.com/science/article/pii/S1361841522002833))
5. Detect outliers based on KNN classifier. ([Arxiv2023](https://arxiv.org/abs/2309.05528))

# Supplementary

All pytorch codes(include submission notebook) are built based on [a simple pytorch-based deep learning framework](https://github.com/m1dsolo/yangdl).
This framework has only a few hundred lines of code and I think it is very suitable for beginners to learn.

1. [submission notebook](https://www.kaggle.com/code/m1dsolo/ubc-ocean-7th-submission)
2. [Training code](https://github.com/m1dsolo/UBC-OCEAN-7th)
