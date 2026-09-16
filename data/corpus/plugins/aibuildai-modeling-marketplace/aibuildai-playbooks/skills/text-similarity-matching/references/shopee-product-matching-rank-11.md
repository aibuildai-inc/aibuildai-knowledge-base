# 11th Place Solution

Competition: shopee-product-matching
Rank: #11
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238181

First of all, I would like to thank all the people who shared their knowledges and suggestions in this competition. I am happy to have won my first gold medal.

The details of my solution are as follows.



# Trainig

I finetuned one image model and two BERT models.

### 1. Image model

* backbone_model : swin_base_patch4_window12_384 (from timm)
* input_size : 384
* loss : ArcFace (scale=34, margine=0.5)
* fc layer (embedding) dimensions : 768
* batch_size = 12 x 2 accumulation (I trained on Google Colab with AMP)
* train epochs : 11
* optimizer : Ranger
* lr scheduler : warmup 2 epochs from 7.5e-6 to 1e-4, cosine decay to 1.5e-5 for backbone, x 2 for fc layer
* augmentations : RandAugment (see blow)

```
from timm.data import create_transform

def get_train_transforms(input_size=384):
    return create_transform(
        input_size=input_size,
        scale=(0.6, 1.0),  # Default: (0.08, 1.0)
        ratio=(1.0, 1.0),  # Default: (3. / 4., 4. / 3.)
        hflip=0.5,
        vflip=0.5,
        is_training=True,
        color_jitter=0.1,
        auto_augment='rand-m3-n1-mstd0.5-inc1',
        re_prob=0.1,  # RandomErasing probability
        re_mode='pixel',  # ['const', 'rand', 'pixel']
        re_count=1,  # number of erasing blocks per image
    )
```


### 2. BERT models

* backbone_model1 : sentence-transformers/paraphrase-xlm-r-multilingual-v1 (from Huggingface)
* backbone_model2 : cahya/distilbert-base-indonesian (from Huggingface)
* loss : ArcFace (scale=30, margine=0.5)
* fc layer (embedding) dimensions : 768
* batch_size = 16
* train epochs :  7 for xlm-r, 8 for indonesian
* optimizer : SAM with AdamW
* lr scheduler : (same as image model, except lr_min)

[This is my BERT training code.](https://www.kaggle.com/shigemitsutomizawa/shopee-training-bert-11th-place-simple-solution)


# Inference


### Prediction Code
```
def get_predictions(df, img_embeddings, bert_embeddings, img_threshold=0.84, bert_threshold=0.84, chunk=32, nearest_one=True, max_preds=42):

    CTS = len(df) // chunk
    if (len(df) % chunk) != 0:
        CTS += 1
        
    preds = []
    for j in tqdm(range(CTS)):
        a = j * chunk
        b = min((j+1) * chunk, len(df))
        img_cts = torch.matmul(img_embeddings, img_embeddings[a:b].T).T
        bert_cts = torch.matmul(bert_embeddings, bert_embeddings[a:b].T).T
        
        for k in range(b-a):
            similarity = (img_cts[k,] / img_threshold) ** 6 + (bert_cts[k,] / bert_threshold) ** 6
            sim_desc = torch.sort(similarity, descending=True)
            
            IDX = sim_desc[1][sim_desc[0] > 1][:max_preds].cpu().detach().numpy()
            o = df.iloc[IDX].posting_id.values
            
            if (len(IDX) == 1) and nearest_one:
                IDX = sim_desc[1][:2].cpu().detach().numpy()
                o = df.iloc[IDX].posting_id.values
            
            preds.append(o)

    return preds
```

[This is my simple inference code.](https://www.kaggle.com/shigemitsutomizawa/shopee-inference-11th-place-simple-solution)

[boundary]
