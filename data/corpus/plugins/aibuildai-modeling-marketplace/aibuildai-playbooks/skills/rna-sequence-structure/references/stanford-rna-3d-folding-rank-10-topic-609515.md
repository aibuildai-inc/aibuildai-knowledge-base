# 10th place solution (How we ensemble)

Competition: stanford-rna-3d-folding
Rank: #10
Source: https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/10th-place-solution-how-we-ensemble

Hi everyone,

I’d like to share our team’s solution for the Stanford RNA 3D Folding competition on Kaggle. Our approach focused on combining three state-of-the-art deep learning models — Protenix, DRFold2, and trRosetta2 — into a robust ensemble pipeline for RNA tertiary structure prediction.

# **Model Structure and Workflow**

**Protenix Implementation**

Diffusion Sampling Parameters: N_sample=10, N_step=150, N_cycle=8
Checkpoint: model_v0.2.0.pt <- The most recent pretrained model weights available on the Protenix GitHub.
Confidence Scoring System: Use in the exact order sorted by the confidence score provided by Protenix.
Hyperparameter optimization did not make a huge difference in my experiments.
I was unable to manage the overfitting problem during Protenix fine-tuning, so I used the pretrained weights for direct prediction. 
Outputs: top 5 PDBs per sample

**DRFold2 Implementation**

FASTA Generation: converts each test sequence to FASTA file format
Filtering: only sequences < 300 nt processed (over 300nt cause timeout failure)
Model Ranking per sample: Reads sel_0 score files -> Selects top 5 models by energy score -> Uses [Arena](https://github.com/pylelab/Arena) for PDB refinement!!
Outputs: top 5 refined PDBs for top predictions (restricted to under 300nt)

**trRosetta2 Implementation(it was hidden in GitHub, not officially published in trRosetta2 article)**

MSA-based prediction for higher accuracy
Model Parameters: nrows=500, refine_steps=0
Filtering: approximately ≤500–600 nt, as far as I recall, though not precise.
Outputs: top 5 refined PDBs for top predictions (restricted to under 500-600nt)


**Ensemble Integration**

Coordinate Extraction: parses PDBs to get C1’ atom coordinates

Merging Strategy:

Protenix: coordinates 1–5 (used as background coordinates)
DRFold2: coordinates 3,4 (overwrite the background coordinates as available as possible)
trRosetta2: coordinates 5 (overwrite the background coordinates as available as possible)

So, final submission.csv is  
|   | 1 | 2 | 3 | 4 | 5 |  
| --- | --- | --- | --- | --- |
| 0~300nt | protenix | protenix | DRFold2 | DRFold2 | trRosetta2 |
| 300~600nt | protenix | protenix | protenix | protenix | trRosetta2 |
| 600nt~ | protenix | protenix | protenix | protenix | protenix |
# 
## CONCLUSION

Ensembling was very effective with three models. However, when I added a fourth model (Boltz), the performance actually decreased. Since I was unable to fine-tune Protenix, I believe others may have succeeded, and I look forward to seeing their solutions.

Last but not least, many thanks to my teammates and Competition hosts!!
