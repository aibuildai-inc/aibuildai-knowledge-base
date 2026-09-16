# ALoRA

LoRA with a per-module rank that moves during training: start every Transformer module with an equal share of a fixed total rank budget, repeatedly score each individual rank by how much the training loss changes when it alone is zeroed out versus kept alone, prune the lowest-scoring ranks, and hand that freed budget to modules that kept all of theirs.

**ALoRA** (Allocating Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) method for large language models, introduced by Liu et al. as an extension of LoRA that "enables dynamic adjustments to the intrinsic rank during the adaptation process" [1]. Its parent is LoRA, which freezes the pretrained weights and injects trainable low-rank decomposition matrices into each Transformer weight, using one fixed rank for every module [2]. ALoRA keeps LoRA's low-rank decomposition but multiplies it by a diagonal gate matrix per module; a routine called AB-LoRA scores each individual rank by the drop in a validation cross-entropy score when that rank alone is masked out, plus the score the rank achieves when it alone is kept active, then periodically prunes the lowest-scoring ranks and reallocates that budget to modules whose ranks were not pruned [1]. The paper is at https://arxiv.org/abs/2403.16187 [1]. It gives two reasons for this design over plain LoRA: the optimal rank varies by task, backbone, and even individual Transformer weight, so one fixed rank per module is not ideal, and prior adaptive-rank methods (AdaLoRA, SoRA, SaLoRA) initialize every module with a rank larger than the target budget - costing extra GPU memory - and depend on heuristic importance scores that the paper argues "may not reliably reflect the contribution of each LoRA rank" [1].

The paper is its own primary evidence of adoption: on LLaMA-2 7B across three GLUE tasks and four question-answering tasks, ALoRA reaches a higher score than LoRA, AdaLoRA, SoRA, and SaLoRA on every one of the seven tasks while ending with fewer trainable parameters (19.6M vs. LoRA's 20.0M initial-and-final), e.g. RTE accuracy 84.6 vs. LoRA's 83.3 and SQuAD f1-em 89.2 vs. LoRA's 88.4 (Table 1) [1]. A citation search on Semantic Scholar returned 58 citing papers as of this check, mostly other rank-allocation or PEFT variants (e.g. "IFCLoRA: Topology-Aware Rank Allocation for Parameter-Efficient Fine-Tuning"), but none was read closely enough here to confirm it names ALoRA's specific documented limitation as the bias it fixes, so no named successor is claimed [3]. Lineage: LoRA (2021 [2]) -> ALoRA (NAACL 2024 [1]); no confirmed named successor.

**When to pick it**: pick ALoRA over plain LoRA [2] when the fixed per-module rank is suspected to be wasting budget on some Transformer weights while starving others, and you can afford periodic pruning-and-retraining rounds during fine-tuning; the paper reports comparable or lower peak training memory than SoRA (18.1GB vs. 18.8GB on LLaMA-2 7B / E2E) because it does not need to over-initialize ranks the way AdaLoRA, SoRA, and SaLoRA do [1]. Its nearest neighbors are exactly those three: ALoRA's related-work section describes AdaLoRA [4] as identifying important ranks with "a sensitivity-based importance score" and describes SoRA [5] as pruning ranks by imposing an $l_0$ norm and optimizing with proximal gradient descent [1]; ALoRA replaces both with the ablation-based AB-LoRA score. There is no offline variant to contrast: ALoRA, like LoRA and its adaptive-rank relatives, trains directly on the downstream task's labeled data, not preference pairs.

**Variant of**: LoRA [2].

**Data it needs**: the same supervised fine-tuning data as LoRA - input/target text pairs for the task, split into a training set and a small validation batch ($B_{val}$, batch size 32 in the paper) used only to compute the AB-LoRA importance scores [1]. The paper's main experiments cover SQuAD, three SuperGLUE tasks (BoolQ, COPA, ReCoRD), three GLUE tasks (SST-2, RTE, QNLI), the E2E generation benchmark, and Alpaca instruction tuning evaluated on MT-Bench, all on LLaMA-2 7B, plus ablations on RoBERTa-large and GPT2-large [1]. Training is on-policy only in the trivial sense that it is standard supervised fine-tuning, not online RL; the "gates" that move rank budget are updated by ordinary backpropagation and by the offline AB-LoRA pruning routine, not by fresh model-generated samples.

**Extra models**: none beyond the base model being fine-tuned - no reward model, value network, or reference model. The AB-LoRA importance score is computed by forward passes of masked or single-rank versions of the same super-network being trained on a held-out validation batch, not a second model [1].

**Shipped by**: no library implements ALoRA. As of this check, HuggingFace PEFT's `PeftType` enum lists `ADALORA` and `GRALORA` but no `ALORA` entry (read from `src/peft/utils/peft_types.py` at HEAD of the `main` branch) [6], and unauthenticated GitHub repository searches for "ALoRA allocating low-rank" (0 results) and "AB-LoRA" (69 results, of which the 30 sampled were all unrelated LoRaWAN-hardware repos) turned up no matching implementation repository [7]. The paper itself reports using HuggingFace Transformers and PEFT to implement all methods in its experiments, but this describes building ALoRA on top of PEFT's LoRA layers for the paper's own experiments, not that PEFT ships ALoRA as a selectable method [1]. Building it would mean extending an existing LoRA layer with a per-rank diagonal gate matrix, plus a separate training-loop routine that periodically freezes the base parameters, masks one rank at a time to score it on a validation batch, prunes, and reallocates - closer to a custom training loop than a drop-in trainer class.

## How it works

The loop in one line: train the current rank allocation for $K_1$ epochs, score every individual LoRA rank by ablation on a validation batch, prune the lowest-scoring ranks and hand their budget to modules that were not pruned, retrain for $K_2$ epochs to recover performance, and repeat for up to $N_A$ rounds [1].

**Forward pass.** For Transformer module $m$ with LoRA rank $r_m$, ALoRA inserts a diagonal gate $G_m$ between the two LoRA matrices:

$$ z = x W_m^A G_m W_m^B, \qquad G_m = \operatorname{diag}(\alpha_{m,1}, \dots, \alpha_{m,r_m}) $$

with $W_m^A \in \mathbf{R}^{d \times r_m}$, $W_m^B \in \mathbf{R}^{r_m \times d}$, and every gate $\alpha_{m,i}$ initialized to 1 [1]. At initialization every module gets an equal share of the target total budget, $r_m^{init} = R^{target}/N_{mod}$, where $N_{mod}$ is the number of tunable Transformer weights per block (7 for LLaMA-2) - unlike AdaLoRA, SoRA, and SaLoRA, which start above the target budget and prune down to it [1].

**AB-LoRA importance score.** For a single LoRA rank $r$, let $M$ be the current super-network, $M_{\backslash r}$ the same network with only rank $r$ zeroed out, and $M_r$ the network with only rank $r$ kept and every other rank zeroed out. With $S(\cdot)$ the negative cross-entropy loss on a fixed validation batch $B_{val}$, the importance score is [1]:

$$ \text{IS}(r) = S(M) - S(M_{\backslash r}) + S(M_r) $$

Since $S(M)$ is constant across ranks being compared, the paper simplifies ranking to $\text{CS}(o) = -S(M_{\backslash r}) + S(M_r)$, using $o$ in this simplified form even though the score is still for rank $r$ [1]. A rank whose removal hurts a lot (low $S(M_{\backslash r})$) or whose lone presence retains most of the performance (high $S(M_r)$) scores as important. The paper chose negative cross-entropy over accuracy or F1 because those metrics may not move when a single rank is masked, and are not well suited to generative fine-tuning [1].

**Pruning and reallocation.** Each round, the $n_A$ (paper default $N_{mod}$, i.e. one per module on average) lowest-scoring ranks are pruned by zeroing their gates; any module left with no pruned rank is judged "important" and receives newly-initialized ranks (added as extra columns/rows to $W_m^A$, $W_m^B$, with gate 1) so the total stays at $R^{target}$; if the number of freed ranks does not divide evenly across un-pruned modules, priority for the extra ranks goes to the modules with the highest average importance score [1]. Worked example from the paper's own budget split: with $n_A = N_A = 8$ freed ranks and three un-pruned modules ranked by average importance, the top two modules each receive 3 ranks and the third receives 2 [1].

**The AB-LoRA score differs from the paper's own earlier DNAS variant, and both differ from AdaLoRA's.** Before proposing AB-LoRA, the paper first frames rank allocation as differentiable neural architecture search (DNAS): the discrete gates $\alpha_i \in \{0,1\}$ are relaxed to $\alpha_i' = 2 \cdot \operatorname{Sigmoid}(a_i')$ with $a_i' \in \mathbf{R}$ so the bi-level objective $\min_\Theta \mathcal{L}(\mathcal{D}_2, \Omega^*, \Theta)$ s.t. $\Omega^* = \arg\min_\Omega \mathcal{L}(\mathcal{D}_1, \Omega, \Theta)$ becomes differentiable [1] (Eq. 4-5). The paper's own ablation (variant "ALoRA-DNAS") uses these learned architecture weights $\alpha_i'$ directly as importance scores instead of AB-LoRA's ablation score, and finds it underperforms AB-LoRA on BoolQ, ReCoRD, and SQuAD [1]. A second ablation ("ALoRA-Sensi") swaps in AdaLoRA's sensitivity-based importance metric and likewise underperforms full AB-LoRA [1].

## Cost

**Theory, from the method's own math**: relative to LoRA, each pruning round in AB-LoRA requires two extra forward passes per rank being scored on the validation batch ($M_{\backslash r}$ and $M_r$), run for up to $N_A$ rounds (paper: $N_A = 8$) - a bounded, one-time-per-round cost rather than a cost added to every training step. No extra trained model is held: gates are scalars per rank, not a value network, reward model, or reference model. Memory beyond LoRA is the gate parameters themselves ($r_m$ scalars per module) plus, transiently, whatever activations are needed to evaluate the masked/single-rank forward passes - far smaller than the weight matrices they multiply.

**In practice, per framework**: not applicable - no framework ships ALoRA (see Shipped by). The paper's own measurement, on its from-scratch implementation built over HuggingFace PEFT for fine-tuning LLaMA-2 7B on the E2E benchmark, reports 18.1GB peak training memory and 5.01 it/s for ALoRA versus 17.6GB / 5.01 it/s for LoRA and 18.8GB / 4.96 it/s for SoRA; total wall-clock training time was 3.81h for ALoRA versus 2.68h for LoRA and 3.63h for SoRA, which the paper attributes to ALoRA needing less memory than SoRA because it does not initialize with a larger-than-target rank (Table 4) [1].

## How to use it

- Data: the same input/target text pairs LoRA needs for the target task, plus a held-out validation batch used only for AB-LoRA scoring (paper: 32 examples) [1].
- Reward/label convention: standard supervised cross-entropy fine-tuning; $S(\cdot)$ in the importance score is defined as negative cross-entropy loss on the validation batch, not accuracy or a task metric [1].
- Key knobs, paper's own values only (no framework ships this method, so there is no framework-default column):

| knob | paper value [1] |
| --- | --- |
| target total rank $R^{target}$ | $8 \times N_{mod}$ (main experiments; swept over $1\times$ to $128\times$ $N_{mod}$ in ablations) |
| initial per-module rank $r_m^{init}$ | 8 |
| ranks pruned/reallocated per round $n_A$ | $1 \times N_{mod}$ |
| max pruning-reallocation rounds $N_A$ | 8 |
| epochs before first scoring round $K_1$ | 1 |
| epochs to recover after each round $K_2$ | 0.25 |
| validation batch size for scoring $B_{val}$ | 32 |
| learning rate (LLaMA-2 7B experiments) | 1e-4, AdamW, linear decay, 6% warm-up |

- Trade-off a run designer faces: a larger $R^{target}$ gives more total budget to move around but costs more memory and compute per forward pass, same as raising LoRA's rank; the paper's own budget sweep (BoolQ and E2E, $R^{target}$ from $1\times$ to $128\times$ $N_{mod}$) states only that ALoRA can "consistently outperform LoRA and SoRA" across budgets, without giving the per-budget numbers in the text of the results section read here (Figure 2) [1]. A larger $N_A$ (more pruning rounds) means more reallocation opportunity but more $K_2$ recovery epochs, i.e. more total training time - directly visible in Table 4's 3.81h vs. LoRA's 2.68h at $N_A = 8$ [1].

## While it runs

- Signals and their healthy shapes: the paper does not publish a training-curve or logging guide for ALoRA runs; the closest first-hand signal is the per-round importance score $\text{IS}(r)$ used to decide pruning, which is by construction the paper's own diagnostic for "which ranks matter" rather than a monitoring metric with a published healthy range [1].
- Published reference runs: Table 1 (LLaMA-2 7B, seven tasks), Table 2 (E2E), Table 3 (Alpaca/MT-Bench), and Table 4 (memory/speed/time) in the paper serve as the only published reference numbers; no separate public training logs were found [1].
- Degeneracies and defaults: the paper's own ablation shows that using the differentiable architecture weights $\alpha_i'$ as the importance score (ALoRA-DNAS) rather than AB-LoRA's ablation score gives worse downstream results, i.e. the "obvious" gradient-based importance signal from the relaxed DNAS formulation is a documented failure mode of a variant, not of ALoRA itself [1]. No config default exists to diverge from, since no framework ships defaults for this method.
- Named successors: none confirmed. A Semantic Scholar citation search for the paper (arXiv:2403.16187) returned 58 citing papers as of this check, several with names suggesting related rank-allocation work, but none was read in enough depth here to state which documented ALoRA bias, if any, it fixes [3].
- Known failure modes: the paper's own Limitations section states only scale and task-coverage limits, not a mechanistic failure mode - it did not test model sizes above LLaMA-2 7B (e.g. 13B, 70B) and did not test task types beyond those in the paper (e.g. information extraction), citing limited compute resources [1]. No GitHub issue tracker exists to check for maintainer-reported numerical traps, since no library ships this method (see Shipped by).
- What the gain is - and is not: the paper's own ablations attribute ALoRA's gain specifically to better rank placement across modules, not to a different LoRA formulation; its rank-allocation visualization on the E2E task shows more of the final budget going to the query and key attention modules and less to the feed-forward layers, which the paper interprets as attention modules needing more task-specific capacity while feed-forward layers already store general language knowledge [1]. The method does not claim to improve on LoRA's expressiveness ceiling, only on how a fixed total rank budget is distributed.

## Sources

[1] Liu, Lyn, Zhu, Tian, Graham, "ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models", NAACL 2024. https://arxiv.org/abs/2403.16187 - defines ALoRA and AB-LoRA, gives the formulation, algorithm, hyperparameters, main results (Tables 1-4), ablations, and Limitations section. Fetched 2026-08-09 (ar5iv HTML full text at https://ar5iv.labs.arxiv.org/html/2403.16187; abstract page fetched separately for the NAACL-2024 acceptance note).

[2] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. https://arxiv.org/abs/2106.09685 - the parent method: freezing pretrained weights and injecting low-rank trainable matrices. Fetched 2026-08-09 (abstract page).

[3] Semantic Scholar Graph API, citations for arXiv:2403.16187. https://api.semanticscholar.org/graph/v1/paper/arXiv:2403.16187/citations - queried for citing-paper titles to check for named successors; returned 58 results, none confirmed read closely enough to state a documented-bias fix. Fetched 2026-08-09.

[4] Zhang et al., "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning", 2023. https://arxiv.org/abs/2303.10512 - cited by [1] as the SVD-sensitivity-based prior work ALoRA compares against. Fetched 2026-08-09 (abstract page).

[5] Ding et al., "Sparse Low-rank Adaptation of Pre-trained Language Models" (SoRA), 2023 - cited by [1] as the $l_0$-norm pruning prior work ALoRA compares against. Cited via [1]; not independently fetched.

[6] HuggingFace PEFT, `src/peft/utils/peft_types.py`, `main` branch. https://github.com/huggingface/peft - `PeftType` enum lists `ADALORA` and `GRALORA` but no `ALORA` entry, checked to confirm no library-level support. Fetched 2026-08-09 (raw file at commit reachable via `main` HEAD on 2026-08-09; unpinned, moving branch).

[7] GitHub repository search for "ALoRA allocating low-rank" and "AB-LoRA". https://github.com - the repository-search queries completed; the first returned 0 results, the second reported total_count 69 of which only the first page of 30 items was fetched and read, and all 30 sampled were unrelated LoRaWAN-hardware repositories (the remaining 39 results were not fetched). A separate attempt to run a GitHub code search for "AB-LoRA" failed with an HTTP 401 authentication error and is not used as evidence here. Fetched 2026-08-09 (GitHub Search API `/search/repositories`, unauthenticated).
