# 6th Place Solution

Competition: stanford-rna-3d-folding
Rank: #6
Source: https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/6th-place-solution

First of all, huge thanks to the organizers and Kaggle for hosting this competition! It’s been a great experience working on the challenge and having such active discussions with everyone.

# Our solution
## Solution overview
Our work is based on AlphaFold 3 [1]. The final model adopts a single-model deep learning based approach. Basically, we leverage the RNA foundation model **AIDO.RNA** [3], integrate its representations into **Protenix** [2] , and finetune the model on the **RNA3DB** database [5].

## Detailed solution
- We augment Protenix with embeddings from AIDO.RNA, a language model pretrained on 42 millions of non-coding RNA sequences.
    - We extract the output hidden states from [AIDO.RNA-650M](https://huggingface.co/genbio-ai/AIDO.RNA-650M) as embeddings using [AIDO.ModelGenerator](https://github.com/genbio-ai/ModelGenerator) [4];
    - We then project the embeddings to the space of single representation of Protenix using a linear transformation.
- We finetune the model with AIDO.RNA frozen on RNA3DB, a well-curated non-redundant RNA 3D structure database with sequence-based clusters. 
    - we use all data in the 2024-12-04 RNA3DB release
    - lr=5e-4, warmup_steps=200, max_steps=10,000, train_crop_size=640, global_batch_size=16
    - exponential moving average (EMA) of model weights with decay rate 0.999
    - no MSAs during training
- For inference, we use the EMA checkpoint saved at the 1600 training step and the default inference setting in Protenix
    - seed=101, n_cycle=10, n_sample=5, n_step=200
    - we use MSAs during inference

## Peformance overview
- Public LB: 0.42849 -> ranked 12th
- Private LB: 0.49758 -> ranked 6th

***

# Takeaways


**Things that didn't work**
- Ensembling: Finetuned Protenix for sequences > 350 nucleotides, DRFold2 [6] for sequences <= 350 nucleotides
     - DRFold2 `cfg_99` was the best checkpoint for us.
     - DRFold2 custom C1 position estimation performed better compared to custom C1 estimation based on P, C4, and N1/N9 atom coordinates.
     - More DRFold2 cycles, 12 seemed to be the best.
- Generate 20 candidate structures, select one as the reference, and use USalign to align the remaining candidates to the reference. Then, calculate the average coordinates.
- We trained a ranker based on pairwise dRMAE and TM-scores between structures. Use map@5 as evaluation metric. LGBM ranker did not produce any significant boost, OOF map@5 = 0.7x for 20 candidates.


# Reference
1. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Google DeepMind. Nature, 2024.* 
2. Protenix-advancing structure prediction through a comprehensive AlphaFold3 reproduction. *ByteDance AML AI4Science Team. bioRxiv, 2025.* 
3. [A large-scale foundation model for rna function and structure prediction. ](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1)*Zou et al. bioRxiv, 2024.* [[Hugging Face Models]](https://huggingface.co/collections/genbio-ai/aidorna-6747516bb48ed96c847f5dd8)
4. [Rapid and Reproducible Multimodal Biological Foundation Model Development with AIDO.ModelGenerator. ](https://www.biorxiv.org/content/10.1101/2025.06.30.662437v1) *Caleb et al. bioRxiv, 2025.* 
5. RNA3DB: A structurally-dissimilar dataset split for training and benchmarking deep learning models for RNA structure prediction. *Szikszai et al. bioRxiv, 2024.* 
6. Ab initio RNA structure prediction with composite language model and denoised end-to-end learning. *Li et al. bioRxiv, 2025.*
