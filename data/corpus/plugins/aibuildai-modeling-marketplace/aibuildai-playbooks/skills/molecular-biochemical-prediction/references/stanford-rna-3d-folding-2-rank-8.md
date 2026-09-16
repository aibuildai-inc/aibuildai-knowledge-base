# 8th Place Solution - TBM and Protenix

Competition: stanford-rna-3d-folding-2
Rank: #8
Source: https://www.kaggle.com/c/stanford-rna-3d-folding-2/writeups/8th-place-solution-tbm-and-protenix

This was my first bioinformatics competition, and I certainly learned a lot, thank you for organizing! And congratulations for the winners!

---

These are the scores of my two selected notebooks. They had only slightly different Protenix computation schedule, and different seeds.
| | Validation | Comp. Public LB | Comp. Private LB | Final Public LB | Final Private LB |
| --- | --- |
| v1 | 0.497 | 0.46411 | 0.57233 | 0.46496 | **0.47641** |
| v2 | 0.493 | 0.46853 | 0.56621 | 0.46828 | **0.47196** |

## Approach Summary

My approach is based on template based modeling (TBM) and [Protenix](https://github.com/bytedance/Protenix). I only used Kaggle resources for this competition (2xT4 GPUs), and I tried to optimize the validation + Public LB score (as my GPU quota allowed). The most important extra details compared to public notebooks are the following:
- TBM:
  - computing secondary structure information using ViennaRNA and incorporating it into sequence alignment;
  - matching the chains independently within a template and solving the linear sum assignment problem (Hungarian algorithm).
- Protenix:
  - for inference of short sequences (length <= 768), splitting chain sequences in JSON, and also the MSA files;
  - increasing the token limit from 512 to 768 using the dynamic chunk size feature of Protenix;
  - parallelizing inference for 2xT4 GPUs, managing time budget, and choosing only the highest scoring candidates;
  - for inference of long sequences (length > 768), only predicting one of each different chains, and treat the others as missing.
- Post-processing and candidate selection:
  - ignoring missing chain coordinates by filling them with downscaled coordinates of other chains (avoiding bad chain alignment penalty);
  - averaging candidates for which (Kabsch aligned) average RMSD is within 2Å,
  - replacing the two lowest scoring TBM candidates by Protenix ones if these TBM candidates do not score over a threshold.

---

## Template-Based Modeling (TBM)

I used [Parasail](https://github.com/jeffdaily/parasail) to align two sequences, and [ViennaRNA](https://github.com/ViennaRNA/ViennaRNA) to calculate the secondary structure information.

### Secondary structure information (SSI)

For all train sequences, I precomputed the SSI in [dot-bracket format](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/io/rna_structures.html). For multi-chain sequences, SSI was calculated for each chain independently. For the sequence alignment, I overlayed the SSI information to the sequences by introducing new characters beyond `"AUCG"` as follows:

     'A' + '.' -> 'A'   ,   'A' + '(' -> 'P'   ,   'A' + ')' -> 'R'   ,
     'U' + '.' -> 'U'   ,   'U' + '(' -> 'S'   ,   'U' + ')' -> 'T'   ,
     'C' + '.' -> 'C'   ,   'C' + '(' -> 'D'   ,   'C' + ')' -> 'E'   ,
     'G' + '.' -> 'G'   ,   'G' + '(' -> 'I'   ,   'G' + ')' -> 'J'   .

For example (target_id: `9I9W`): `"GGCACUGGAAGUGCGGCACUGGAAGUGC" + ".(((((...))))).(((((...)))))" -> "GIDPDSGGARJTJEGIDPDSGGARJTJE"`.

Then I replaced the 4x4 distance matrix by a 12x12 one for sequence alignment to score the SSI alignment simultaneously. The sequence aligner configuration included the scores for the sequence match and mismatch (`pmatch` and `pmismatch`), the ssi match and mismatch (`smatch` and `smismatch`), and the gap opening and extending penalties (`gap_open` and `gap_extend`). I used the following two configuration to enhance diversity and selected the five best (aggregated) candidates in a round robin way (where two of which could be replaced later by Protenix):

    pmatch: 30,  pmismatch: -20,  smatch: 10,  smismatch: -10,  gap_open: 95,  gap_extend:  4,
    pmatch: 30,  pmismatch: -20,  smatch:  1,  smismatch:  -1,  gap_open: 60,  gap_extend: 10.

Parasail uses integer scores, hence I used larger numbers compared to biopython. More importantly, Parasail expects capital letters in the alphabet by default, an issue which I was lucky to accidentally figure out from its documentation instead of almost discarding the whole idea. The `smismatch` penalty was doubled for aligning incorrect bracket SSIs (letters related to `'('` and `')'`, and vice versa).

I picked candidates from two different configurations because sometimes using the SSI information could yield worse quality templates (e.g., `9G4R` and `9KGG`), and also the different gap structure seemed to provide extra diversity in the candidates. I only used global alignments (`nw_` prefixed Parasail functions). In order to sort templates, I recalculated the alignment scores ignoring tail gaps in each chains. Still, I could not find a reliable unified way to pick candidates from multiple configurations at once, hence I used round robin.

I calculated SSI based on the sequence itself without using the MSA information. This is quite fast to compute with ViennaRNA. I also tried to use the consensus-based SSI using the MSA data, but it can take significant time to ViennaRNA to compute (hours for longer sequences). As all attempted runs to use this instead of the cheap no-MSA SSI resulted in worse score for me, I abandoned this idea. 

### Multi-chain alignment

Based on [this discussion](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/discussion/680551), I wanted a matching algorithm which does not rely on the somewhat arbitrary chain ordering of multi-chain RNA sequences. Therefore, I aligned all chains between the query and a template sequence (N x M alignments for an N-chain query and an M-chain template). The chain mapping was chosen to be the solution to the linear sum assignment problem (as implemented in [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)).

I had maintained a big expectation for this feature, but to my surprise, it did not really move the score much. It was kept though, because I really did not like to depend on the provided chain ordering. Although it feels like a lot of computation, Parasail is very fast, and the whole TBM part of my notebook takes only a few minutes to finish overall.

---

## Protenix

In order to reduce noise, I performed deterministic Protenix inference using `protenix.utils.seed.seed_everything(seed, deterministic=True)`. If I did not misunderstand the Protenix code, the `use_rna_msa` switch has no effect without the `use_msa` switch, which further needs `unpairedMsaPath` to be specified in the input JSON file. Eventually, I turned on all of these MSA switches along with the `use_template` switch as well.

Protenix inference can produce poor quality results, so I tried to generate as many candidates as time allowed, sort them by the `ranking_score` of Protenix, and pick the 1-2 best as needed. The cheapest config in the [Protenix paper](docs/PTX_V1_Technical_Report_202602042356.pdf) used 5 seeds and 5 diffusion samples resulting in 25 candidates, which was already way beyond the time budget here, at least I could not scale up Protenix that much.
I used the GPU T4 x 2 accelerator on which I ran two independent Protenix instances in parallel with the same parameters, but with different seeds. At the end, I aggregated the candidates, sorted them, and used the best 1-2 of them to replace the worst 1-2 TBM candidates. Occasionally, I skipped Protenix for a sequence if at least five templates with sufficiently high scores were found during TBM.

I fed sequences at once into Protenix up to length 768 using the dynamic chunk size setting (related to the attention / pairformer blocks), reducing the chunk size from the default 256 to 128 for sequences of length between 512 and 768, which allowed the inference to fit into the 16GB of VRAM.
In this case, I split the sequence and the corresponding MSA file per chain to separate `rnaSequence` records in the JSON file.

For sequences longer than the 768 character limit, I predicted the chains independently. Chains being still longer than the 768 limit were split up with overlaps (of size 256), and their predictions were aligned using the Kabsch algorithm, similarly as has been shown by other publicly available notebooks.
In this case, I only predicted the different chains once, and if there were chain copies, I left them as missing.

Instead of leaving the chain copies missing, I tried to copy over the predicted chain and align them using either the best templates from TBM, or by an extra inference run from Protenix which was predicting the sequence at once with all chains being truncated. Unfortunately, all of these attempts ended up with worse scores instead of just leaving these copies as missing. I believe, these methods did not capture the alignment correctly, and USalign was penalizing more such chain misalignments than the missing chains (which I hided "inside" another chains, see below more on this).

I implemented a safeguard which exits the Protenix loop as the 8 hours run-time limit approaches. For sequences of size below 768, I allowed at most 5 inference runs with different seeds up to a 5 or 6 minutes limit per GPU in my v1 or v2 notebooks, respectively. For longer sequences, I only allowed a single run. In all cases, the first run generated as many diffusion samples as was requested based on the quality of the TBM scores, while on subsequent runs I only calculated a single diffusion sample. As I noticed, extra diffusion samples were almost as expensive to compute as completely new samples with a fresh seed.

---

## Post-processing and candidate selection

The TBM candidate scores usually lay between 0 and 1, but they could sometimes exceed 1 (e.g., `9G4J`) due to the SSI scores (my normalization did not account for this).

I implemented various constraint-based post-processing steps based on atom distances, but I did not observe any measurable improvement from using them.

### Filling in missing coordinates

This was the only useful post-processing step I found, and therefore the only one I used. At the end, I applied a standard linear interpolation/extrapolation step to fill in coordinates. However, before doing so, I checked for completely missing chains, which I handled differently. If an identical non-missing chain was available (which was often the case), its coordinates could be simply copied over. I found that it worked slightly better to shrink the copied coordinates toward their average position, which is the approach I ultimately used. I believe it helps because the filled-in chain coordinates do not distort the global alignment in USalign. As a result, the alignment of the original non-missing chains improves, leading to a slightly better score.

### Averaging candidates

To diversify the predictions, I aligned the candidates using the [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm) and averaged those with an average [RMSD](https://en.wikipedia.org/wiki/Root_mean_square_deviation) of no more than 2Å.

### Candidate selection

I selected at most five candidates from TBM after averaging. I applied very light filtering (e.g., score >= 0), which usually filled all five candiadate slots. I then replaced up to two of the lowest-scoring TBM candidates with Protenix candidates. The exact number (0-2) depended on the TBM scores (using two thresholds: 0.5 and 0.8) and the number of distinct Protenix candidates after averaging. The v1 notebook runs the entire Protenix inference loop again until it times out (limit: 8 hours - 10 minutes).

---

## Acknowledgements

The most influential notebooks for me and what I learned from them:
- MMseqs+TBM+Parasail: https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-templates
- Protenix: https://www.kaggle.com/code/sigmaborov/stanford-rna-3d-folding-top-1-solution
- Evaluation: https://www.kaggle.com/code/rhijudas/mmseqs2-3d-rna-template-identification-part-2

I also thank ChatGPT for its endless patience of talking through many ideas and writing me some code.

## What did not work

I implemented, but abandoned the following ideas:
- using MSA consensus-based ViennaRNA secondary structure (consistently worse);
- templates based on MMsegs search (no improvement);
- post-processing the coordinates based on distance constraints and trajectory smoothing (no improvement or worse);
- orienting independent chain predictions based on either templates or truncated multi-chain Protenix predicted candidates (consistently worse)
