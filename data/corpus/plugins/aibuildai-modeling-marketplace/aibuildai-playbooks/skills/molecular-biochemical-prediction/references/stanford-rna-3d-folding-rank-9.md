# 9th place solution  - d4t4 team

Competition: stanford-rna-3d-folding
Rank: #9
Source: https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/10th-place-solution-d4t4-team

**Many thanks to Kaggle and the competition hosts for organizing this great challenge and giving us the chance to take part.**
### 📂 Dataset Creation

We built our training dataset by merging several complementary sources:

- **Stanford RNA 3D Folding Competition Data**
  - Versions: v1 and v2 (v2 used as the primary training set).

- **CASP16 Top Predictions (Pseudo-labels)**
  - We included Top-1 predicted structures per target as pseudo-labels.

- **RNA PDB Database (cutoff: March 31, 2025)**
  - Only entries released on or before March 31, 2025 were included.

---

### 🧩 Handling Missing Residues
- Residues with unresolved atoms in PDB were kept in the sequence but masked in coordinates.
- Missing atoms were zero-padded or ignored in loss computation.
- This ensures reliable training without losing sequence context.

---

### 🔁 Multiple Conformations
- For some datasets (PDB and CASP16), we generated up to 5 conformations per RNA sequence.
- These conformations were stored as (num_conf, seq_len, 3) arrays in the dataset.
- During training, we supported two strategies:
  1. Top-1 only → using the first conformation as a deterministic baseline.
  2. Multi-conf sampling → exposing the model to all 5 conformations, either:
     - Returned as coordinate_multi (all conformations available at once), or
     - Randomly selecting one conformation per epoch, so the model sees different structures across training.
- This approach increased structural diversity and improved generalization without changing the overall dataset size.

---

### 🔬 MSA Pipeline
- For sequences where evolutionary information was available, we incorporated Multiple Sequence Alignments (MSA) into training.
- We used MMseqs2 to build RNA MSAs against a curated RNA sequence database.
- Precomputed MSAs were stored in a dedicated directory and linked during training.
- For sequences without MSA coverage, the pipeline fell back to dummy/no-MSA features, ensuring consistent input formats.
- This hybrid strategy enriched structural context for many targets and improved accuracy on conserved RNAs.

---

### 📊 Final Outputs
- Consolidated merged sequence dataset across competition, CASP16, and PDB.
- Aligned atom-level labels with missing residues masked and up to 5 conformations per sequence.
- Precomputed MSAs (via MMseqs2) for a subset of structures.

===========================================================

## ⚙️ Training Configuration

- **Hardware**
  - GPU: NVIDIA H100 (96 GB)
  - Mixed precision: bfloat16 (bf16)

- **Optimization**
  - Batch size: 8
  - Max steps: 12,000
  - Warmup steps: 50
  - Learning rate: 1e-4
  - Crop size: 800 nucleotides

- **Sampling**
  - Diffusion steps: 20

- **Evaluation**
  - Checkpoint interval: every 2,000 steps
  - Evaluation interval: every 50,000 steps

- **Features**
  - MSA pipeline: precomputed with MMseqs2 for sequences where alignments were available.
  - Fallback mode: dummy/no-MSA features used when MSA coverage was not available.
  - Multi-conformations: up to 5 conformations per RNA sequence included.
  - Masked residues: unresolved residues in PDB masked to avoid noisy supervision.

===========================================================

##  Conformation Selection Strategy

- Generate multiple conformations per sequence.  
- Step 1: Select Top-1 structure by pLDDT score (highest confidence).  
- Step 2: From the remaining predictions, iteratively select conformations that maximize RMSD diversity using the Kabsch RMSD algorithm.  
- Final output: Top-5 diverse conformations per sequence.  

===========================================================

## 📊 Results Summary

| Model / Strategy                                     | Sequence Length | Training Steps | Score   |
|------------------------------------------------------|-----------------|----------------|---------|
| **Protenix** (standard)                              | ≤ 800           | 8000           | 0.46388 |
| **Protenix + RibonanzaNet** (hybrid, >800 handled)   | ≤ 800 + >800    | 8000           | 0.478   |
| **Protenix + Nufold** (hybrid, >800 handled)   | ≤ 800 + >800    | 8000           | 0.479   |

- **Submission**: Single-model, no ensemble → **0.46388** on leaderboard.
