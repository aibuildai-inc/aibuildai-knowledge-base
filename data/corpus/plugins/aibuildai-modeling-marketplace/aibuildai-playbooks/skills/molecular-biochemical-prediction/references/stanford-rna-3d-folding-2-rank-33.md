# 33rd Place Solution

Competition: stanford-rna-3d-folding-2
Rank: #33
Source: https://www.kaggle.com/c/stanford-rna-3d-folding-2/writeups/33rd-place-solution

# Stanford RNA 3D Folding Part 2 — 33rd Place Solution (Silver Medal)

**Final Submission: ex77 Adaptive Diversity Selection**

| | |
|---|---|
| **Best Private LB** | 0.447 |
| **Best Public LB** | 0.433 |
| **Private LB Rank** | 33rd / 1,877 (Silver Medal) |

---

## Solution: Overall Approach

Our solution **ex77 Adaptive Diversity Selection** is an ensemble pipeline that combines 3 structure prediction methods and **adaptively determines the optimal slot allocation per target**.

- **Phase 1: TBM (Template-Based Modeling)** -- Template matching based on sequence similarity
- **Phase 2a: DRfold2** -- Deep learning prediction for short-chain RNA
- **Phase 2b: Protenix** -- Diffusion model-based structure prediction
- **Phase 3: Adaptive Selection** -- Diversity maximization via Kabsch RMSD + FPS

### Pipeline Diagram

[pipeline]
*Figure 1: Solution Pipeline -- 3-method ensemble + adaptive diversity selection*

### Phase 1: TBM (Template-Based Modeling)

Template-based modeling is the **core component** of this solution. It uses known structures from training data as templates and transfers 3D coordinates based on sequence alignment.

#### Sequence Alignment

Global alignment using BioPython's `PairwiseAligner`:

| Parameter | Value | Note |
|-----------|-------|------|
| match_score | 2 | Match bonus |
| mismatch_score | -1.5 | Mismatch penalty |
| open_gap_score | -8 | Gap opening |
| extend_gap_score | -0.4 | Gap extension |

#### Template Adaptation

The `adapt_template_to_query` function processes alignment results:

- **Match positions**: Directly transfer template 3D coordinates
- **Gap positions**: Linear interpolation from flanking known coordinates
- **Terminal gaps**: Extrapolation at 3.0A intervals from nearest known coordinates

#### Template Pool

Integrated `train_sequences` + `validation_sequences` (~5,700 sequences). Generates **top_n=30** candidates per target.

### Phase 2a: DRfold2

**DRfold2** is a deep learning structure prediction model specialized for short-chain RNA (100nt or less).

- Target: Only sequences with length <= 100nt
- Model: cfg_97 configuration
- Time limit: 2 hours
- Output: Extract C1' atom coordinates from PDB files

DRfold2 predictions are added to the diversity pool and utilized in Phase 3 FPS selection.

### Phase 2b: Protenix

**Protenix** is an AlphaFold-based diffusion model structure prediction tool that runs inference on all targets.

| Parameter | Value |
|-----------|-------|
| N_SAMPLE | 5 |
| SEED | 42 |
| MAX_SEQ_LEN | 512 (absolute limit) |
| CHUNK_OVERLAP | 128 |
| USE_RNA_MSA | true |
| USE_MSA | false |
| USE_TEMPLATE | false |

> **MAX_SEQ_LEN=512 is the absolute limit** -- Setting it to 600 causes OOM on Kaggle T4 GPU (16GB VRAM), resulting in scoring failure. Sequences exceeding 512 are handled via chunking.

### Phase 3: Adaptive Diversity Selection

The **key innovation** of this solution is the mechanism that adaptively determines TBM slot count per target.

- **top pct_id >= 50%**: TBM 3 slots + diversity pool 2 slots
- **top pct_id < 50%**: TBM 0 slots + diversity pool 5 slots

The diversity pool selects the most diverse structures from all DRfold2 + Protenix candidates using **Farthest Point Sampling (FPS)**.

[adaptive]
*Figure 2: Adaptive slot allocation logic*

#### Farthest Point Sampling (FPS)

Algorithm to maximize diversity:

1. Compute **Kabsch RMSD matrix** between all candidates
2. Select the candidate with the largest average RMSD first
3. Select the candidate with the maximum minimum RMSD to already-selected candidates
4. Repeat until the required number of slots is filled

This maximizes the structural space covered by the 5 predictions, improving the Best-of-5 TM-score.

### Chunking Strategy

For long RNA sequences exceeding MAX_SEQ_LEN=512 (e.g., 9ZCC=1460nt, 9MME=4168nt), we use **overlapping chunk splitting**.

[chunking]
*Figure 3: Chunking strategy for long RNA sequences*

#### Stitching (Assembly)

Procedure to combine Protenix outputs from each chunk into full-length coordinates:

1. Compute optimal rotation/translation via Kabsch alignment (SVD decomposition) on overlap regions
2. Align subsequent chunk coordinates to the previous chunk
3. Use **linear blending** (weighted average) on overlap regions for smooth connection

### RNA Physical Constraints

The `adaptive_rna_constraints` function applies physical plausibility to each structure:

| Constraint | Parameter | Description |
|-----------|-----------|-------------|
| Bond distance | 5.95 A | Ideal distance between adjacent C1' atoms |
| Next-nearest distance | 10.2 A | Ideal distance between C1' atoms 2 residues apart |
| Laplacian smoothing | 0.06 | Local coordinate smoothing |
| Clash avoidance | 3.2 A | Minimum distance between non-adjacent residues |
| Passes | 2 | Number of constraint application iterations |
