# 1st Place Solution

Competition: stanford-rna-3d-folding
Rank: #1
Source: https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/1st-place-solution

Thank you Kaggle and the competition hosts, for this incredible competition and the opportunity to participate. Your passion for this challenge was truly infectious and served as one of my driving forces throughout the competition. 

Since this is my first gold and my first win on Kaggle, I would like to take the opportunity to thank @jhoward for the fast.ai course, @radek1 for their book which was an inspiration in my ML journey. Khan Academy for helping me rethink mathematics, and @huggingface for their amazing deep learning courses. 


## Competition Strategy

My approach was clear from the outset. Without GPUs, training a model from scratch or fine-tuning was not viable. My early research - drawing on CASP results, literature, and conference talks, including one by host @rhijudas - showed that Template-Based Modeling approaches consistently dominated. Based on this, I committed to TBM from day one and spent the next 90 days refining my method.

Next, I focused on the evaluation metric, since understanding it determines the exploration path. TM-score has two key properties: it is normalized by structure length (so 50nt and 200nt RNAs are compared on the same 0-1 scale), and it is robust to local errors - a small number of misplaced nucleotides does not disproportionately lower the score. This insight allowed me to prioritize getting the overall fold correct over achieving atomic-level precision.


## Data Strategy and Model Selection

The host-provided dataset was comprehensive. I systematically processed all CIF files in the provided PDB_RNA directory with comprehensive nucleotide mapping (93 variants including modified bases) and disorder-aware coordinate extraction. This process ensured complete coverage of the available structural data, capturing modified nucleotides that standard parsers might otherwise miss.

After exploring nearly all available open-source models, I selected DRfold2 as the optimal choice due to its extensive potential for optimization. Rather than fine-tuning the model itself, I focused on enhancing its optimization and selection modules. This strategy improved prediction quality while ensuring the pipeline could execute efficiently on Kaggle GPUs.

## Template-Based Modeling (TBM)

TBM follows a five-step process:

**1. The Search - Finding Similar Structures**

The goal is identifying database structures that resemble the target sequence through sequence alignment.


**2. The Alignment - Sequence Mapping**

This step creates the translation guide between query and template using global sequence alignment with gap penalties optimized for RNA. 


**3. The Transfer - Coordinate Inheritance**

Straightforward copying of 3D coordinates for all matched positions. This leverages the evolutionary tendency for RNA to conserve 3D structure more than sequence. 


**4. The Gap Fill - Geometric Backbone Reconstruction**

For insertions and deletions, I relied on geometric principles maintaining RNA's characteristic backbone:

- Maintains `C1'-C1'` distance (`~5.9Å` between consecutive nucleotides)
- For compressed gaps: extends the backbone with realistic curvature using sinusoidal perturbations perpendicular to the backbone direction
- For normal gaps: uses linear interpolation between flanking coordinates
- Terminal extensions follow the established backbone direction


**5. Adaptive Refinement - Confidence-Based Optimization**

The refinement intensity adapts to template confidence score:

- High-confidence templates (>0.8 similarity): minimal constraints, preserving template geometry
- Medium-confidence templates: moderate sequential distance constraints (`5.5-6.5Å`)
- Low-confidence templates: additional steric clash prevention and light base-pairing constraints. 
- Constraint strength scales as: `0.8 × (1 - min(confidence, 0.8))`



## DRfold2 Enhancements

### Selection Module

- **Double Precision Calculations:** Consistent float64 operations reduce numerical errors for more reliable model rankings
- **Vectorized Distance Calculations:** GPU-accelerated pairwise distance computation via torch.cdist
- **Optimized Energy Functions:** Pre-computed cubic spline coefficients enable fast structure scoring without repeated spline fitting

These improvements were motivated by the authors' own observations that DRfold2 sometimes failed to select its best models. For example, they report cases where the 5th-ranked model significantly outperformed the top-ranked one (p. 8, lines 305–317), underscoring the need for more robust post-processing and ranking protocols (p. 9, lines 319–321). 

My modifications directly targeted this weakness by making scoring and ranking more accurate and consistent.



### Optimization Module

- **PyTorch LBFGS:** Native optimizer with automatic differentiation delivers more accurate gradients and better convergence than SciPy implementations.
- **GPU Acceleration:** Energy calculations and gradient computations performed on GPU where possible.
- **External Knowledge Integration:** Enhanced capabilities through Boltz-1 integration (credit to @youhanlee) - (2nd notebook submission)

The authors themselves highlight the flexibility of DRfold2's optimization framework, demonstrating this by integrating AlphaFold3 conformations as an additional potential term (p. 9, lines 327–329). This hybrid approach yielded significantly better results than either method alone, achieving higher TM-scores and lower RMSDs (p. 9, lines 331–334). They conclude that such integration represents a promising direction for future improvements (p. 10, lines 370–372). 

My own optimization experiments followed this spirit of extensibility, focusing on GPU acceleration and integration of Boltz-1.  



## Hybrid Strategy

The final pipeline uses a strategic combination:

- **Template-based modeling:** For shorter sequences and when time budget is exhausted. 
- **DRfold2:** For the rest of sequences where deep learning excels. 
- **Graceful fallback:** DRfold2 failures automatically fall back to template approach.


Special shoutout to @hengck23 for consistently sharing valuable research papers, open-source models, and insights that served as invaluable resources for the community.
