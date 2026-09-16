# 6th place solution

Competition: shopee-product-matching
Rank: #6
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238010

First of all, we'd like to appreciate the hosts for giving us such a great opportunity to learn. 
We'd also like to appreciate great participants who shared their thoughts and knowledge, 
especially @cdeotte and @ragnar123 made a great contribution to the competition. Thank you!

# Summary
[model_strucure]
[ensemble]

<br>

# Models
|               | #1                            | #2                             | #3                             | #4                          |
|---------------|-------------------------------|--------------------------------|--------------------------------|-----------------------------|
|            cv |                         0.877 |                          0.875 |                          0.877 |                       0.873 |
|      img_size |                           224 |                            224 |                            560 |                         560 |
|  img_backbone | swin_large_patch4_window7_224 |  swin_large_patch4_window7_224 |                efficientnet_b3 |             efficientnet_b3 |
| text_backbone |              xlm-roberta-base | bert-base-multilingual-uncased | bert-base-multilingual-uncased |           bert-base-uncased |
|        cnn_lr |                          4e-5 |                           4e-5 |                           2e-4 |                        2e-4 |
|       bert_lr |                          1e-5 |                           1e-5 |                           5e-5 |                        5e-5 |
|         fc_lr |                          5e-4 |                           5e-4 |                           2e-4 |                        2e-4 |
|    batch_size |                            16 |                             16 |                             16 |                          16 |
|     scheduler |   linear_schedule_with_warmup |              ReduceLROnPlateau |    CosineAnnealingWarmRestarts | CosineAnnealingWarmRestarts |
|          loss |          ArcFace(m=0.5, s=32) |           ArcFace(m=0.5, s=32) |           ArcFace(m=0.5, s=50) |        ArcFace(m=0.5, s=50) |
|     optimizer |       AdamW(weight_decay=0.1) |                           Adam |                           Adam |                        Adam |

<br>

# Similarity
Use euclidean distance and cosine similarity to get nearest neighbors.
We use threshold for submission, 


<br>

# Threshold adjust
CV+0.2 for cosine similarity threshold and CVx0.7 for euclidean distance.

<br>

# Ensemble Method
4 models * 3 output(concat, text, img) * 2 distances = 24 votes


<br>


# PostProcess
## Force prediction one to two
If there is one prediction, we made it two.
By doing this, we increased CV/LB score by +0.01

<br>


## Remove unmatch units
* thanks to discussion -> https://www.kaggle.com/c/shopee-product-matching/discussion/229495
* We removed the data that have unmatched units. For Example:

```

prediction data title: "Anmum Lacta 400gr" 

-----------------------
[no_postprocess]: 
Anmum Lacta 400gr
Anmum Lacta Menyusui 400gr
Anmum Materna 200gr Cokelat
Anmum Emesa 200Gr
Anmum Emesa Chocolate 200 gr
Anmum Lacta 200gr Cokelat
-----------------------
[postprocess]: 
Anmum Lacta 400gr
Anmum Lacta Menyusui 400gr
-----------------------
[target]: 
Anmum Lacta 400gr
Anmum Lacta Menyusui 400gr
-----------------------

<< f1 >>: 0.5 -> 1.0
```

<br>



# validation
* groupKfold(5), group by label_group

<br>


# worked for us
* torch.cat(img_embeddings, text_embeddings) -> nn.Linear -> arcface
* output 3 embeddings (concat, image, text) [LB + 0.01]
* voting ensemble
* different learning rate for cnn/bert/fc

<br>


# not worked for us
* multimodal approach(image + text)
  * transformer
  * convolution
* ensemble
  * averaging cosine similarity
  * concatenate embedding -> calculate cosine similarity
* pretrain cnn and bert respectively
* bert
  * CLS layer
* cnn
  * efficientnet_b4, b5, b6, or larger models
  * gem(pooling)
* loss
  * AdaCos
  * SphereFace
  * TripletLoss
  * AdaptiveFace
* activation layer(Swish, GeLU, ...)
* large batch size(>=32)
* progressive learning
* tf-idf for ensemble
