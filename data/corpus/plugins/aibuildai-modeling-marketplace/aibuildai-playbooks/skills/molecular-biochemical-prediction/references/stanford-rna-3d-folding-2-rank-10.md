# 10th Place Solution

Competition: stanford-rna-3d-folding-2
Rank: #10
Source: https://www.kaggle.com/c/stanford-rna-3d-folding-2/writeups/10th-place-solution

First word: thanks the host for organize this competition and kagglers for their contribution. This solution absolutely would not have been possible without the incredible contributions of the Kaggle community, so a huge thank you to everyone who shared their insights and baselines. 

---
I definitely expected a big shake-up in this competition, but I honestly didn't expect to rank this high! 

The toughest part of this competition for me was figuring out how to properly validate and diversify the 5 required predictions. To finalize my submission, I meticulously tracked both my local validation score (`validation_sequences.csv`) and the Public LB score. 

Here is a breakdown of the core implementation that led to this result.

### 📌 TL;DR
My final pipeline relies on an ensemble of:
* **TBM** + **Protenix** + **RNAPro**
* **Chunking** long sequences specifically for Protenix.

The 5 submission spots were allocated as follows:
* **2 spots for TBM**
* **1 spot for Protenix**
* **2 spots for RNAPro**

My two final selected submissions used the exact same logic, but with one key difference for diversity: one had `USE_RNA_MSA=false`, and the other had `USE_RNA_MSA=true`.

---

## Section 1: Inference Baseline

### 1. TBM (Template-Based Modeling)
The TBM logic builds upon several great public notebooks. Merging the validation and training sets to act as a larger template pool for searching when submitting only. train set only when running local validation

**The Pipeline:**
1. **Search:** Scan up to `top_n = 30` candidates using a global `PairwiseAligner`. Rank these by their normalized alignment score: $$\text{norm\_score} = \frac{\text{aln.score}}{2 \times \min(\text{len\_query}, \text{len\_template})}$$  and percent identity. Filter the results by `MIN_SIMILARITY` and `MIN_PERCENT_IDENTITY`.
2. **Adapt:** Map template coordinates to the query using alignment-aware copying alongside interpolation/extrapolation (`adapt_template_to_query`). This ensures chain segments and stoichiometry are respected so multi-chain templates map correctly.
3. **Produce Predictions (TBM = 2):**
   * **Slot 1:** Adapted template (straight copy).
   * **Slot 2:** Adapted template + small perturbation Gaussian noise with 

$$\sigma = \max(0.01, (0.40 - \text{sim}) \times 0.06)$$ .If similarity is very low, a geometric perturbation (hinge/jitter) is used instead.
4. **Refine:** Apply soft geometric smoothing via `adaptive_rna_constraints()` to keep local geometry realistic. 

**Key Parameters:**
* `top_n=30`
* `adaptive_rna_constraints` passes=2
* `apply_hinge` uses $$\text{deg} \approx 22$$

### 2. Protenix
* **Quality Control:** Extract the C1' coordinates robustly. Reject collapsed or degenerate outputs (e.g., near-zero inter-residue distances), and pad or repeat valid samples as needed so every target ends up with the required number of predictions.
* **Long-sequence Strategy:** Long RNAs were split into overlapping segments. Each segment was inferred independently, and the full-length structure was rebuilt by aligning overlapping regions and smoothly blending coordinates to ensure a continuous backbone. I used `MAX_SEQ_LEN ≈ 512` and `CHUNK_OVERLAP ≈ 64`. 

There were only 2 samples in the validation set that required chunking (`9MME`, `9ZCC`), but the chunking approach showed consistent local improvement over non-chunking:

| Target | No Chunking | With Chunking |
| :--- | :--- | :--- |
| **9MME** | ~0.1376 | **0.1855** |
| **9ZCC** | ~0.2320 | **0.2781** |

### 3. RNAPro
I passed the TBM-derived predictions directly to RNAPro as structural templates for refinement. RNAPro was run with MSA + RibonanzaNet2 embeddings, 10 recycling cycles, and 200 diffusion steps.

---

## Section 2: How I Came Up With This Solution

This was the hardest part. I ran inference on the 28 validation samples repeatedly with different settings. Here is what I discovered:
* Protenix predictions are non-deterministic, but generating multiple Protenix predictions for a single runtime didn't yield much helpful diversity.
* RNAPro performed worse on the Public LB but was performing *very well* locally. 
* Setting RNA_MSA to True greatly enhance local validation score by `~0.03` but LB scores bellow 0.4 when combining with TBM.

Here is a log of my tracked runs:

| Model / Strategy | Local Validation | Public LB | Private LB | Final Private LB |
| :--- | :--- | :--- | :--- | :--- |
| 2 TBM - 1 Protenix - 2 RNAPro <br> *(with RNA_MSA False for Protenix)* | - | [0.399;0.424] | [0.565;0.586] | 0.474 |
| 2 TBM - 1 Protenix - 2 RNAPro <br> *(with RNA_MSA True for Protenix)* | - | [0.400;0.414] | [0.578;0.588] | |
| 2 TBM - 2 Protenix - 1 RNAPro <br> *(with RNA_MSA True for Protenix)* | - | [0.395;0.397] | 0.578 | 0.471 |
| 2 TBM - 2 Protenix - 1 RNAPro <br> *(with chunking long sequences for Protenix)*| 0.45 | [0.399;0.424] | [0.565;0.586] | |
| 3 TBM - 2 Protenix <br> *(With RNA_MSA on)* | 0.47 | 0.401 | 0.540 | |
| TBM + reserve 2 spot for Protenix | 0.43 | [0.409;0.419] | [0.560;0.590] | |
| TBM + reserve 1 spot for Protenix | 0.44 | [0.407;0.419] | [0.572;0.586] | |
| main TBM + Protenix to fill leftover spot <br> *(my public notebook)* | 0.42 | [0.396;0.429] | [0.503;0.537] | |
| RNAPro (seq < 1000) + fill with TBM | 0.47 | 0.357 | 0.489 | |
| pure Protenix | | 0.249 | 0.503 | |
| pure TBM | 0.36 | 0.368 | 0.458 | |

**The Turning Point:**
The first completed Protenix inference [notebook](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm) (which I published) had a critical bottleneck: it prevented making Protenix predictions entirely if 5 TBM predictions passed the threshold. 

After fixing this to ensure Protenix samples were explicitly generated, my local validation improved by `0.02`. However, there was no significant Public LB improvement. My hypothesis was that the Public LB was heavily dominated by the TBM approach. And many later public notebooks that included Protenix didn't fix this bottleneck either. 

---

## Section 3: What Didn't Work (Or What I Gave Up On)

* **Reranking Protenix by Model Confidence:** Creating a large pool of Protenix samples and reranking them based on the model's own confidence scores just cost way too much GPU time for too little return.
* **Finetuning Protenix:** This was slightly out of my current comprehension level. I did try to preprocess and cache a small subset of data, but I ultimately ran out of time to actually get a training run working.
* **Using Templates for Protenix:** I already set up on my public notebook to use template successfully, but I observed absolutely no difference on the Public LB and my local validation scores.

---

## References:
* https://www.kaggle.com/code/amirrezaaleyasin/openrnafold-starter3
* https://www.kaggle.com/datasets/qiweiyin/protenix-v1-adjusted
* https://www.kaggle.com/code/alexxanderlarko/protenix-v1
* https://github.com/bytedance/Protenix
* https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-rnapro-inference

---

Thanks again to everyone, and congratulations to all the winners!
