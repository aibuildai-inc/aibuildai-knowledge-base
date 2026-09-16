# Public 13th / Private 16th solution 💪

Competition: shopee-product-matching
Rank: #16
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238128

Shopee is my first kaggle competition from start to finish, and also the competition I worked hardest. In this challenge, images, text and post-processing are both important factors for achieving a good result, therefore it is difficult to optimise all them at once. Fortunately, my teammates @meliao @ywenlu @dandingclam @mathurinache provided great help.

Also, congratulations to @ywenlu for becoming a competition master !

## 1. Summary
Our main solution is based on chris' famous baseline and ragnar's arcface training notebook. 
We spent more than half of the time focusing on the optimization of one single image model and TF-IDF but blocked at around 0.740.

Until two weeks before the end of the competition, we finally discovered the importance of post-processing. Because we found that nearly 4000 rows did not match any other products by observing local matches, and this case does not exist in the target.

With the help of pp, we achieved 0.763 using only a B3 and TF-IDF. Then we ensembled other models to achive 0.771 on public.



## 2. Image
Based on ragnar's stratified dataset, we use 1/3 data to train and check the val loss on the other 2/3. This helped us to quickly identify useful augmentations and parameters. I published here my results : [Shopee Image Benchmark](https://www.kaggle.com/vicioussong/shopee-image-benchmark)

Below the models we used at final stage :

| Model                     | Image size | Augmentations                                                                                                                                                      | Val loss |
|---------------------------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| efficientnet b3           | 512        | central_crop<br />random_crop<br />random_flip_left_right                                                                                                                  | 14.2     |
| eca_nfnet_l1              | 512        | RandomResizedCrop<br />CenterCrop<br />RandomBrightnessContrast<br />OneOfBlur                                                                                                 | 14.31    |
| efficientnet b4 cosine LR | 384        | central_crop<br />random_crop<br />random_flip_left_right<br />random_hue<br />random_saturation<br />random_contrast<br />random_brightness                                             | 14.34    |
| efficientnet b4           | 456        | central_crop<br />random_crop<br />random_flip_left_right<br />random_flip_up_down<br />random_hue<br />random_saturation<br />random_contrast<br />random_brightness<br />mosaic_augmentation | 14.3     |

## 3. Text
### 3.1 TF-IDF
**Pre-processing**
- remove emojis, punctuations, e-commerce stop-words (like 'ready', 'stock', 'free', 'gift', 'jaring', 'sabun', 'siap', 'kirim', 'diskon', I spent 3 hrs to identify them manually)
- lemmetization using English & Indonisian lemmetizer (Indonisian lemmetizer can be found here [shopee_simplemma](https://www.kaggle.com/vicioussong/shopee-simplemma/), source : https://github.com/adbar/simplemma/)
- split number and unit : 150ml => 150 ml
- split attached n-grams : fresh care => freshcare

**Feature**
I am not sure this is a good choice but I did use all features including single caracters : 

```
TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", binary=True)
```

### 3.2 Sbert
Used public kernel [Reaching 0.612 with Text Only : Shopee](https://www.kaggle.com/tanulsingh077/reaching-0-612-with-text-only-shopee)

## 4. Post-processing

### 4.1 Strategies
1. Recall 50 neighbours using KNN 51 and filter using the rules below :
    - if img_score > th_top
    - if txt_score > th_top
    - if img_score > th_img and txt_score > th_min
    - if txt_score  > th_txt and img_score> th_min
    - if avg(img_score, txt_score  ) > th_mean
2. Recall the closest neighbour using KNN 2 based on img_score + txt_score. If one row predicts only itself, we replace its result to this.
3. Closest neighbour propagation
    - Considering the great improvement of step 2, we propagate the new matched items
4. Label propagation with a strict threshold KNN 51 on rows where nb_match=2
    - if A-B and B-C then A-B-C

### 4.2 OOM problem solving
Instead of concatnating image embeddings with text embeddings using numpy, we converted them to torch tensor and use the following fonction to calculate the cosine similarity to avoid OOM problem.

```
def text_knn_2(df, tensor1, tensor2, K, th, chunk = 128):
    out_preds = []
    for i in tqdm(list(range(0, df.shape[0], chunk)) + [df.shape[0]-chunk]):
        arr = tensor1[i : i + chunk] @ tensor1.T + (tensor2[i : i + chunk] @ tensor2.T)
        if len(df) > 3: 
            indices = torch.nonzero((arr > th) & (arr >= arr.sort(descending=True).values[:,K-1].reshape(arr.shape[0],-1)))
        else:
            indices = torch.nonzero(arr > th)

        preds = dict()
        for k in range(arr.shape[0]):
            preds[k] = []
        for ind in range(indices.size(0)):
            preds[indices[ind, 0].item()].append(indices[ind, 1].item())

        out_preds.extend([(df.iloc[k].posting_id, df.iloc[v].posting_id.tolist()) for k, v in preds.items()])
    return out_preds[:df.shape[0]]
```

Thanks for all Kagglers, I have learnt a lot in this competition.

Hope to see you again in the next journey 😏
