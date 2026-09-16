# 19th Place Solution

Competition: stanford-rna-3d-folding-2
Rank: #19
Source: https://www.kaggle.com/c/stanford-rna-3d-folding-2/writeups/19th-place-solution

# 19th Place Solution

---

## Dear Organizers

Thank you for hosting and organizing this competition.  
I spent three months working on this task, and it was a highly meaningful and rewarding experience.

---

## Overview of the Approach

### ■ Models

- **Fine-tuning**  
  Fine-tuning was applied to Protenix and RNAPro.  
  Due to GPU memory limitations, training was restricted to relatively short sequences.

- **TBM (Template-Based Modeling)**  
  - **Conventional TBM**  
    We used the TBM from the public notebook [1].  
  - **VFold-like motif-based TBM**  
    Based on the Vfold3D paper [2], we constructed a motif/loop library and applied a TBM approach that assembles structures using secondary structures predicted by SPOTRNA.

- **Protenix Inference Settings**  
  Inference was performed both with and without MSA.  
  Additionally, weights from different training steps were used to generate diverse predictions.

---

### ■ Refinement with RNAPro

- TBM predictions were passed to RNAPro for refinement.  
- Masked templates were used, and each template was processed independently.

---

### ■ Prediction Selection

- All predictions were ranked based on an internal score, and the top results were selected to fill the five submission slots.

---

### ■ Methods Tried but Not Included in the Final Submission

The following approaches showed some improvements but did not significantly impact the public score, and were therefore not included in the final submission:

- **Automatic structure selection via clustering and ranking**  
  - Performed 4-seed inference using Protenix V1 (each seed outputs 5 structures)  
  - The resulting 20 structures were clustered using RMSD and ranked using PARSEbp [3]  
  - Structures were automatically selected based on this process  
  - This approach improved the public score by **+0.3**, but was not included in the final submission

- **Multimer-specific model**  
  Although this model occasionally produced structures with high TM-scores, we were unable to develop a reliable method to consistently select those structures, so it was not included in the final submission.

---

## References
- [1] https://www.kaggle.com/code/kami1976/stanford-rna-3d-enhanced-model
- [2] https://pmc.ncbi.nlm.nih.gov/articles/PMC9728534/  
- [3] https://www.biorxiv.org/content/10.1101/2025.10.13.682106v1.full
