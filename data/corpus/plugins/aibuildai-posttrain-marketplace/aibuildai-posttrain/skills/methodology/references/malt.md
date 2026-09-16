# MALT

Train three copies of a base model into a Generator, Verifier, and Refiner by sampling a branching search tree over training questions, propagating outcome rewards backward through the tree via value iteration, and post-training each role separately with SFT (and DPO for the verifier and refiner) on the resulting correct/incorrect pairs.

**MALT** (Multi-Agent LLM Training) is a post-training strategy that splits reasoning into a sequential pipeline of a Generator, a Verifier, and a Refiner, introduced by Motwani et al. as "a novel post-training strategy that divides the reasoning process into generation, verification, and refinement steps using a sequential pipeline of heterogeneous agents" [1]. All three roles start from the same base model and are trained with supervised fine-tuning (SFT); the paper treats SFT as its baseline single-agent building block, over which it layers Direct Preference Optimization (DPO) for the verifier and refiner [1][2]. For the generator's role, the paper explicitly frames its SFT step as "analogous to standard STaR post-training... with a specialized dataset" [1], making STaR [3] the closest single-agent precedent it names for that step. The paper is at https://arxiv.org/abs/2412.01928 [1]. The mechanism: for each training question, sample n generator outputs, then n verifier critiques per generator output, then n refinements per critique, producing $n^3$ trajectories per question; label each final refinement correct or incorrect against the ground-truth answer, then use value iteration to propagate that binary reward backward so every intermediate generator and verifier output also gets an expected value; binarize at a 0.5 threshold and build correct-vs-incorrect preference pairs per role [1]. The paper gives three motivations: end-to-end gradient propagation through multiple discrete-token agent outputs is hard, so training happens per-role instead; sparse outcome-only rewards make credit assignment across agents difficult, which the value-iteration step is built to solve; and a single model handling generation, verification, and refinement together limits exploration and self-correction compared to specialized roles [1].

MALT was accepted at COLM 2025 [1] and reports gains on Llama-3.1-8B-Instruct of a relative 15.66%, 9.40%, and 7.42% over the untrained single-agent baseline on MATH, CSQA, and GSM8K respectively, taking raw accuracy from 49.50%/74.50%/84.25% to 57.25%/81.50%/90.50% (Table 1) [1]. The paper itself reports no landmark adopters; it is a recent (submitted 2024-12-02), independently confirmed-by-title arXiv paper, and at the time of writing its own project page lists the code as "Code (Coming Soon)" with no adopting systems named [1][4]. Lineage in one line: STaR (2022 [3]) and DPO (2023 [2]) -> MALT (2024 [1]), with the paper's own appendix proposing PPO-based online multi-agent training as a stated future direction, not a shipped successor [1].

**When to pick it**: pick MALT when a task has a checkable ground-truth answer (math, commonsense QA) and the goal is a specialized generate-verify-refine pipeline rather than a single model, and when synthetic role-specific preference data can be generated offline via search before any training happens [1]. Prefer plain DPO [2] on a single model when there is no need to decompose the task into roles. Prefer plain STaR [3] when only positive-sample bootstrapping is wanted and negative/preference data is not needed; the paper found STaR-only (SFT) baselines underperform MALT and, for the generator specifically, found DPO does not help over SFT alone because Llama-3.1-8B-Instruct is already heavily preference-tuned as a generator [1]. The paper names online multi-turn RL agent training (e.g., ArCHer [5]) as the live-interaction alternative to MALT's offline scheme, and separately notes AgentQ's [6] MCTS-based approach as the source of the DPO-optimal-policy theorem MALT's credit assignment relies on [1].

**Variant of**: DPO [2], applied per-role on top of an SFT stage the paper describes as analogous to STaR [3][1].

**Data it needs**: a training set of `<question, ground-truth answer>` pairs only - the paper used GSM8K (7.47k train), MATH (7.5k train), and CSQA (9.74k train) [1]. No human-labeled role-specific data is required; the tree-search + value-iteration procedure generates SFT and DPO (chosen/rejected) pairs for each of the three roles automatically, yielding roughly 2k-6k labeled pairs per model per benchmark at branching factor $n=3$ [1]. Offline: all training data comes from a fixed search tree collected before post-training begins, not from fresh online sampling during training [1].

**Extra models**: none beyond the three role copies of the same base model (Generator, Verifier, Refiner), which start identical and diverge only through per-role LoRA fine-tuning [1]. No separate reward model - the reward is a deterministic exact-match check against the ground-truth answer [1] - and no frozen reference policy beyond the standard DPO reference (the SFT-updated version of each role, per the paper's own DPO setup) [1][2]. Cost details below.

**Shipped by**: no framework this card checked implements MALT as a named trainer or recipe. As of this writing the paper's own project page marks its code release "Coming Soon" [4]; a GitHub repository search found only a static project-page repository under one of the authors, with no training code [7]; and a keyword search of trl's documentation table of contents and verl's documentation index found no mention of MALT [8]. This covers trl and verl only, not every post-training framework. Building it requires two different jobs on top of an existing SFT/DPO trainer (e.g., trl's `SFTTrainer`/`DPOTrainer`): a new n-ary tree-sampling loop over Generator/Verifier/Refiner outputs, and a value-iteration/binarization step to turn the tree into per-role preference datasets before any existing trainer runs [1].

## How it works

Loop in one line: sample a branching tree of Generator -> Verifier -> Refiner outputs per question, score only the leaves against ground truth, then average scores back up the tree to label every node, and train SFT (+ DPO where used) per role on the resulting labels [1].

**Tree sampling** [1]: for question $q_i$, sample $n$ generator outputs $\{g_{i,j}\}_{j=1}^n$; for each $g_{i,j}$, sample $n$ verifier outputs $\{v_{i,j,k}\}_{k=1}^n$; for each $v_{i,j,k}$, sample $n$ refiner outputs $\{r_{i,j,k,l}\}_{l=1}^n$. This produces $n^3$ trajectories per question.

**Leaf reward** [1]:

$$ \mathcal{V}(r_{i,j,k,l}) = \mathcal{R}\big(T(r_{i,j,k,l}), a_{GT}\big) \in \{0, 1\} $$

where $T$ extracts the final answer from the refiner's output and $\mathcal{R}$ is 1 if it matches the ground truth $a_{GT}$, else 0 [1].

**Backward value propagation** [1], a Monte Carlo average of child values:

$$ \mathcal{V}(v_{i,j,k}) \approx \frac{1}{n}\sum_{l=1}^{n}\mathcal{V}(r_{i,j,k,l}), \qquad \mathcal{V}(g_{i,j}) \approx \frac{1}{n}\sum_{k=1}^{n}\mathcal{V}(v_{i,j,k}) $$

**Binarization** [1]: any node is labeled correct if its value exceeds 0.5, incorrect otherwise:

$$ \hat{s} = \checkmark \text{ if } \mathcal{V}(s) > 0.5, \text{ else } \times $$

**Worked example**, $n=3$: suppose one generator output $g_{i,j}$ produces three verifier outputs, and each verifier output's three refinements score $\{1,1,0\}$, $\{1,0,0\}$, and $\{0,0,0\}$ respectively against the ground truth. The three verifier values are $\mathcal{V}(v_{i,j,1})=2/3\approx0.67$ (labeled $\checkmark$), $\mathcal{V}(v_{i,j,2})=1/3\approx0.33$ (labeled $\times$), $\mathcal{V}(v_{i,j,3})=0$ (labeled $\times$); a correct-vs-incorrect DPO pair for the verifier can be built from $v_{i,j,1}$ against either $v_{i,j,2}$ or $v_{i,j,3}$. The generator value is $\mathcal{V}(g_{i,j})\approx(0.67+0.33+0)/3\approx0.33$, labeled $\times$ despite one of its downstream branches reaching a fully correct refinement - a partial-credit signal the paper argues a plain binary outcome label on the generator alone would miss [1].

**Training per role** [1]: SFT is run on all three roles using only the positive (correct-labeled) examples from each role's dataset $D_G$, $D_V$, $D_R$; DPO is then run only on the SFT-updated Verifier and Refiner, using chosen/rejected pairs from $D_V$ and $D_R$, with the reference policy being each role's own SFT checkpoint [1]. The Generator is SFT-only: the paper reports DPO lowers Generator performance (e.g., 52.25% -> 51.25% on MATH) because Llama-3.1-8B-Instruct's own post-training already applied DPO on a similar generator data distribution, making a second DPO pass prone to overfitting [1]. The DPO loss follows the standard form [2]:

$$ \mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}_{(x,y^+,y^-)\sim \mathcal{D}_{train}^{DPO}}\;\sigma\left(\beta\log\frac{\pi_\theta(y^+|x)}{\pi_{ref}(y^+|x)} - \beta\log\frac{\pi_\theta(y^-|x)}{\pi_{ref}(y^-|x)}\right) $$

with $\beta=0.2$ in the paper's runs [1]. Appendix A.6 gives a theoretical justification, adapting a theorem from AgentQ [6], that optimizing this DPO objective on the binarized value labels approximates the optimal RL policy $\pi^*(a|h_t)\propto\pi_{ref}(a|h_t)\exp(\hat{Q}(h_t,a)/\beta)$ under the binarized value function [1].

## Cost

**Theory, from the method's own math**: data generation needs $n^3$ trajectories per training question ($n=3$ gives 27 in the paper's runs) [1], each trajectory requiring one Generator, one Verifier, and one Refiner forward generation - so total generation calls scale as $|D_{train}|\cdot(n + n^2 + n^3)$ across the three stages. Because a Generator output is shared across its $n^2$ downstream Verifier/Refiner branches and a Verifier output across its $n$ Refiner branches, the calls at each stage can be batched and parallelized against a shared input, which the paper notes keeps the branching factor from adding much wall-clock overhead at data-collection time [1]. Training itself does not scale with $n^3$: after the tree is collected, each role trains once on its own fixed labeled dataset (2k-6k pairs per role per benchmark) [1]. At inference MALT is not a tree search: it runs one sequential Generator -> Verifier -> Refiner pass, repeated for a fixed number of self-consistency votes (3 in the paper), independent of the training-time branching factor $n$ [1].

**In practice, per framework**: no framework this card checked ships MALT (see Shipped by), so there is no framework-level cost report to cite. The paper's own practical mitigation is LoRA: all three roles are fine-tuned with LoRA adapters on the shared base model rather than full fine-tuning, so "LoRA adapters ensure that the model weights themselves aren't duplicated, thus requiring only minimal additional memory for the adapters themselves while the base models remain the same" [1]. The paper does not state the LoRA rank, GPU type/count, or wall-clock time used for training or data generation [1].

## How to use it

- Data prep: only `<question, ground-truth answer>` pairs are needed; the paper used GSM8K, MATH, and CSQA question/answer pairs directly, with no human annotation of intermediate generator/verifier/refiner steps [1].
- Reward convention: exact-match of the extracted final answer against ground truth, binary $\{0,1\}$; there is no learned reward model or judge in the main method [1].
- Role-conditioning: each of the three roles uses a fixed prompt template distinguishing its function (generate / verify / refine), reused identically across baselines and MALT-trained models [1].
- Key knobs, with the paper's own values as the only source (no framework anchors exist):

| knob | paper value [1] |
| --- | --- |
| branching factor $n$ | 3 (27 trajectories/question) |
| SFT learning-rate multiplier | 0.1 |
| SFT batch size | 8 |
| DPO learning-rate multiplier | 0.1 |
| DPO $\beta$ | 0.2 |
| DPO adapter weight | 0.2 |
| epochs | varied 1-10 per model/benchmark based on dataset size |
| binarization threshold | 0.5 |
| inference self-consistency votes | 3 |
| sampling temperature (Llama-3.1-8B-Instruct) | 0.3 |

- Trade-offs a run designer faces: raising $n$ grows the training-data tree cubically ($n^3$) for a linear increase in per-stage sampling diversity, so the paper caps it at 3 for compute reasons rather than reporting an ablation over $n$ [1]. The paper separately ablates removing a role: a two-agent Generate+Refine (skipping Verifier) pipeline scored 54.75%/84.75%/76.25% and Generate+Verify (skipping Refiner) scored 55.75%/88.75%/78.00% on MATH/GSM8K/CSQA, both below the full three-agent pipeline's 57.25%/90.50%/81.50% (Table 5) - the deciding comparison for whether all three roles are worth keeping [1].

## While it runs

- Signals and their healthy shapes: the paper does not report training-time logging metrics (loss curves, KL, entropy) for MALT's SFT/DPO runs; the only "signals" it reports are held-out accuracy and turn-wise accuracy (how much each of Generator, Verifier, and Refiner improves the running answer), shown to increase monotonically across turns on the benchmarks tested [1].
- Published reference runs: the paper's own Table 1 is the reference curve - MALT reaches 57.25%/81.50%/90.50% on MATH/CSQA/GSM8K with Llama-3.1-8B-Instruct, versus 49.50%/74.50%/84.25% for the untrained single-agent baseline and 52.25%/76.25%/81.75% for an SFT-only (STaR) baseline [1]; no other public reference logs were found by this card, since no framework or public run it checked has reproduced it (see Shipped by).
- Degeneracies and defaults: a group of one ($n=1$) collapses the search tree to a single deterministic path per stage, giving no correct/incorrect contrast to build DPO pairs from - the paper does not run this configuration, so this is a construction-level inference from the pairing rule in Sec. 4.3, not an observed result [1]. Uniform-reward branches (all $n$ children of a node score the same) yield no usable preference pair for that node under the paper's pairing scheme, which requires one $\checkmark$ and one $\times$ child sharing the same parent [1].
- Named successors: none found by this card. A Semantic Scholar lookup for arXiv:2412.01928 returned only a citation count (66 as of this check), not a list of citing titles, so no citing-paper search was actually performed and no successor could be checked that way [9]. The paper's own appendix proposes PPO-based online multi-agent training with the same credit-assignment scheme as a future direction, not a published successor [1].
- Known failure modes: the paper's own limitations section notes that even at low sampling temperature, benchmark performance shows high variance, which is why results are averaged over four seeds on random test-set subsets [1]. It separately reports, as an ablation finding rather than a general limitation, that DPO degrades Generator performance specifically on an already DPO-post-trained base model (Llama-3.1-8B-Instruct), attributed to overfitting on a similar data distribution [1]. No GitHub issue tracker exists to check for maintainer-reported failure modes, since no code has been released (see Shipped by) [4][7].
- What the gain is - and is not: the paper positions the gain as better use of inference-time role specialization and search-derived credit assignment, not new base-model capability - it explicitly frames MALT as complementary to, not competing with, single-model reasoning-RL approaches like DeepSeek-R1, and suggests its search trajectories could instead be distilled into instruction-tuning data to warm-start a single model before online RL [1]. It does not claim MALT teaches the base model new knowledge beyond reorganizing how existing capability is deployed across the three roles.

## Sources

[1] Motwani, Smith, Das, Rafailov, Laptev, Torr, Pizzati, Clark, Schroeder de Witt, "MALT: Improving Reasoning with Multi-Agent LLM Training", published as a conference paper at COLM 2025, arXiv:2412.01928 (v3, 2025-10-06). https://arxiv.org/abs/2412.01928 - defines MALT: tree sampling, value iteration, binarization, SFT/DPO training per role, experimental results, ablations, appendix algorithm and hyperparameters, limitations. Fetched 2026-08-09 (HTML full text via arxiv.org/html, and PDF text via arxiv.org/pdf, both v3).

[2] Rafailov, Sharma, Mitchell, Ermon, Manning, Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - DPO objective. Fetched 2026-08-09 (abs page).

[3] Zelikman, Wu, Mu, Goodman, "STaR: Bootstrapping Reasoning With Reasoning", NeurIPS 2022. https://arxiv.org/abs/2203.14465 - cited by [1] as the analogue for the Generator's SFT step. Fetched 2026-08-09 (abs page).

[4] MALT project page. https://multiagentllmtraining.com/ - lists code as "Coming Soon" and links paper/HuggingFace/slides, no adopting systems named. Fetched 2026-08-09.

[5] Zhou, Zanette, Pan, Levine, Kumar, "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL", 2024. https://arxiv.org/abs/2402.19446 - cited in [1]'s related-work discussion as the live-interaction online RL contrast to MALT's offline scheme. Fetched 2026-08-09 (abs page).

[6] Putta, Mills, Garg, Motwani, Finn, Garg, Rafailov, "Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents", 2024. https://arxiv.org/abs/2408.07199 - source of the DPO-optimal-policy theorem adapted in [1]'s Appendix A.6.2. Fetched 2026-08-09 (abs page).

[7] GitHub repository search for "MALT multi-agent LLM training"; the only match, chansmi/MALT-Improving-Reasoning-with-Multi-Agent-LLM-Training, is a static project-page site (JavaScript, GitHub Pages content only), not a training-code release. https://github.com/chansmi/MALT-Improving-Reasoning-with-Multi-Agent-LLM-Training. Repository metadata read via the GitHub API on 2026-08-09 gave default_branch `master`; querying `GET /repos/.../commits/master` at that time resolved it to commit `b7905dcd07f55e47df5158797c9fa7ac4baede04` (committed 2025-03-10T15:31:25Z). The "no training code" claim is pinned to the root-tree file list at that commit, which contains only `.DS_Store`, `.github/`, `.nojekyll`, `CNAME`, `README.md` (blob sha db62023c683d509c919e14ebdf789d8435fde420), `index.html`, and `static/` - no training scripts.

[8] trl documentation table of contents (`docs/source/_toctree.yml`, `main` branch) and verl documentation index (`docs/index.rst`, `main` branch), both grepped for "malt"/"multi-agent"/"multi_agent" with zero matches. https://github.com/huggingface/trl and https://github.com/volcengine/verl - unpinned, mutable `main`-branch docs; read 2026-08-09.

[9] Semantic Scholar paper record for arXiv:2412.01928, citationCount 66 at time of check. https://api.semanticscholar.org/graph/v1/paper/arXiv:2412.01928 - the fetched response contains only a citation count field, no list of citing paper titles; no citing-paper search was performed. Fetched 2026-08-09 (API JSON).
