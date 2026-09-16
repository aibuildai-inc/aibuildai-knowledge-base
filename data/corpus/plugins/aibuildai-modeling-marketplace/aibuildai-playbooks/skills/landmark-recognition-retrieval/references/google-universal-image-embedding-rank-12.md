# 12th Solution

Competition: google-universal-image-embedding
Rank: #12
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359497

Thank to host for hosting a great competiton, hard work with my team member ( @hycloud ) and competitors.
It's great competition. I have learned many things from this competition and solutions. that's very happy.  

We write our 12th solution as follows.

## Dataset
Data 1: 
- GLDv2-Clean
- Met-Art
- Products-10k
- Shopee-Products
- Food-1k
- DeepFashion
- Standford-Cars
- Storefronts (shared public kaggle dataset)
- Furniture (shared public kaggle dataset).

Data 2:
- GLDv2 Full
- Met-Art
- Products-10k
- Shopee-Products
- Public Dataset
- rp2k
- product10k
- deepfashion
- fashion 200k
- myautoge
- Storefronts (shared public kaggle dataset)
- ifuniture
- stanford_online_shop

## Model
|Model|Loss|Neck|Data|Score(Private/Public)|
|----|----|----|----|----|
|ViT-Huge-224|Arcface with s=30|1024-4096-BN-PReLU-64|Data1|0.629/0.609|
|ViT-Huge-224|Multi Similarity Loss + CrossBatchMemory|1024-4096-BN-PReLU-64|Data2|0.624/0.622|
|Swin-Large-384|Arcface with s=30|1536-6144-BN-PReLU-64-BN-PReLU|Data1|0.612/0.602|
|Swin-Large-224|Arcface with s=30|1536-6144-BN-PReLU-64-BN-PReLU|Data1|0.576/0.562|

Freeze backbone and train only projector for ViT-H; Two-stage for Swin, first stage freeze backbone second stage fine-tune whole model; 10 to 20 epochs for first stage, 1 to 5 epochs for second stage.
Optimizer: SGD, lr 0.1 for first stage, 0.001 for second stage.(ViT-H used Adam)

# Ensemble
We used ensemble for a final layer using PCA.
We pickup competition distributed dataset for PCA dataset. that's important point. 
final result is pickup PCA dataset > full dataset.

Final Score is 0.667/0.654(Private/Public).

# Didn't worked
- Stable Diffusion(storefront generation
- CLIP 2stage training(Backbone freeze->full)
