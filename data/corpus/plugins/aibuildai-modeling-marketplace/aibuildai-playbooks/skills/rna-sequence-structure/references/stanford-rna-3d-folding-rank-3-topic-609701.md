# 3rd Place Solution

Competition: stanford-rna-3d-folding
Rank: #3
Source: https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/3rd-place-solution

# Our 3rd Place Solution – Stanford RNA 3D Folding

First of all, thanks to Kaggle and the organizers for putting together such a challenging and interesting competition.  

We finished **3rd on the public leaderboard** before the unseen data release, and also **3rd on the private leaderboard**.  

Earlier we shared a short overview of our solution. This post goes into more detail.  

---

## Summary of Our Solution
- Ensemble of **DRfold2**, **Protenix**, and **Boltz-1**  
- Fine-tuned **Protenix** with newly released RNA datasets  

---

## Detailed Solution

### Dataset

- **rMSA**  
  We generated our own rMSA data using the [official rMSA code](https://github.com/pylelab/rMSA) provided by the hosts.  
  - The v2 rMSA released by the organizers did not cover the full set of recently published data, so we had to build our own.  
  - This process took about **14 days**, even with multiprocessing and several servers.  
  - Our rMSA data is available here: [Google Drive](https://drive.google.com/drive/folders/15bhXWsR6QuDQo6U4j8Ii8-OE4B4SmC7i?usp=drive_link).  

- **Training dataset**  
  We tested two versions:  
  1. **Full RNA dataset** – included all recently available data, such as [CASP16](https://www.kaggle.com/datasets/tant64/casp16) uploaded by @tant64. To check quality, we compared some of the labels with the [CASP16 GitLab repository](https://gitlab.com/arneelof/CASP16-predictions/-/tree/main/rna_results?ref_type=heads).  
  2. **RNA-only dataset** – excluded complexes (RNA bound to proteins/DNA), following the clarification from the hosts [here](https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/575745).  

Both versions of the training data and labels are available here: [Google Drive](https://drive.google.com/drive/folders/1XKYzk2oCcHPt6DB_wLNYL-7w3s1R793n).  

---

### Models

We built our ensemble using models from these official repositories:  
- [DRfold2](https://github.com/leeyang/DRfold2)  
- [Protenix](https://github.com/bytedance/Protenix)  
- [Boltz](https://github.com/jwohlwend/boltz)  

We initially explored new architectures, but given the time and resource limits, it was more effective to focus on fine-tuning and combining existing models.  

---

#### DRfold2
- Performed best on sequences **<400 nt** with v1 data.  
- Running the full pipeline with optimization, clustering, and all 80 released checkpoints was too slow.  
- We found that **energy selection + Arena** gave most of the gain, while the other steps added little to the TM-score but took significant runtime.  
- So our DRfold2 setup used **energy selection + Arena** only for the post-processing.  

---

#### Protenix
- Trained on an **NVIDIA GH200 (96GB VRAM)**.  
- Limited to sequences **<800 nt** due to memory issues.  
- Used fine-tuning code from @lihaoweicvch ([discussion link](https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/573495)).  
- Fine-tuning **without rMSA** did not help, so we trained **with rMSA**.  
- Only modified `max_steps`; a full dataset run took about a day.  
- Early checkpoints (2–3 cycles) performed worse; performance improved with longer fine-tuning.  
- We also tested **multiple Protenix outputs → DRfold energy selection → Arena**, but it did not improve results.  
- For inference, we modified [@geraseva’s code](https://www.kaggle.com/code/geraseva/protenix) to include rMSA.  

---

#### Boltz-1
- While not the strongest model on its own, including **one Boltz prediction** improved ensemble diversity.  
- This helped because the scoring metric favored picking the best out of multiple diverse outputs.  
- We used [@youhanlee’s inference notebook](https://www.kaggle.com/code/youhanlee/boltz-1-inference-submission).  

---

### Final Submissions & Code

We are sharing our original submission notebooks. Some commented-out code remains, which shows parts of our experiment history.  

1. **[DRfold2 + Protenix](https://www.kaggle.com/code/yekim102/drfold4-pro-msa-pro-msa-base-rna)**  
   - Public LB: **0.60338** | Private LB: **0.52787**  
   - <400 nt: 3 × DRfold2 + 2 × Protenix (RNA-only, with MSA)  
   - >400 nt: 2 × Protenix (RNA-only) + 2 × Protenix (All) + Protenix baseline  

2. **[Protenix + Boltz](https://www.kaggle.com/code/yekim102/fin-pro-all-1-rna-2-base1-boltz1)**  
   - Public LB: **0.61253** | Private LB: **0.54312**  
   - 2 × Protenix (RNA-only) + Protenix (All) + Protenix baseline + Boltz baseline  

---

## Acknowledgements

Many thanks to the community members who shared resources and insights:  
- **@hengck23** – for extensive discussions and guidance.  
- **@geraseva** – for early Protenix inference code.  
- **@lihaoweicvch** – for Protenix fine-tuning code.  
- **@youhanlee** – for updated Boltz inference code.  

Finally, thanks to **Eigen Company** for providing resources for this competition.
