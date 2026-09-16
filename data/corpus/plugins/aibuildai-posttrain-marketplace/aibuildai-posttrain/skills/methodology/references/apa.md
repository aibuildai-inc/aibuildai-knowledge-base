# APA

Fine-tune a model to answer questions it already handles correctly and to ask for clarification on the rest, where "the rest" is picked by an entropy-drop signal the model computes about its own uncertainty, not by ground-truth ambiguity labels.

Item home: https://arxiv.org/abs/2404.11972 (v3, the revision fetched for this card [1]).

**APA** (Alignment with Perceived Ambiguity) is a four-stage supervised-fine-tuning (SFT) pipeline for open-domain QA, introduced by Kim et al., who describe it as a method that leverages an LLM's own assessment of ambiguity to fine-tune the LLM to handle ambiguous queries, detecting and managing them appropriately [1]. The paper positions APA against two existing families of LLM alignment, RLHF and SFT [1], and its own limitations section names RLHF (via InstructGPT [2]) and DPO [3] as alternative alignment methods it does not use [1]; its closest sibling by mechanism is data-quality-driven SFT such as LIMA, which the paper's related-work and results discussion cite for the principle that alignment quality depends on data quality over quantity [1]. APA's four stages are: (1) Initial Prediction Assessment - run the model on labeled unambiguous and ambiguous queries and sort outputs into a correct set and an incorrect set; (2) Perceived Ambiguity Detection - for incorrect samples, have the model self-disambiguate the query and flag it as perceived-ambiguous if that self-disambiguation measurably reduces the model's own output entropy; (3) Response Construction - build a clarification-request target for each perceived-ambiguous sample, either from a fixed phrase list or model-generated; (4) SFT - fine-tune on a class-balanced mix of the correct set and the perceived-ambiguous set with standard next-token cross-entropy [1]. Two reasons the paper gives for this design: standard LLM training does not specifically teach a model to handle ambiguous input, and the degree of ambiguity an LLM perceives depends on the knowledge it possesses, so a fixed, human/gold ambiguity label does not necessarily match what a given model needs to be trained on [1].

APA was confirmed by an exact-title arXiv search on "Aligning Language Models to Explicitly Handle Ambiguity," which returned this paper as the unique, top-cited match against a name-collision candidate ("Adaptive Pluralistic Alignment," 2 citations) that shares only the APA acronym and is unrelated in mechanism. The paper reports no landmark adopters; it explicitly notes a lack of directly comparable prior work and compares only against its own inference-only and trained baselines [1]. Its own gains: on Llama2-7B, training on 3,088 perceived-ambiguous-plus-correct samples (APA-Fixed) beats a Full-set SFT baseline trained on all 10,036 labeled samples on both metrics on most of the five evaluation datasets, e.g. SituatedQA-Geo F1_a 41.86 (APA-Fixed) vs. 41.45 (Full-set) vs. 0.00 (Direct prompting) (Table 1) [1]. Lineage in one line: standard SFT / instruction alignment -> data-quality-selected SFT (LIMA, 2023 [4]) -> APA (2024 [1]), with no named successor found in a citation search of this paper (69 citations at the time of sourcing) as of 2026-08-09.

**When to pick it**: pick APA when you have a QA dataset with gold (un)ambiguous labels and gold answers/clarification targets, and you want a model that answers confidently on unambiguous queries but asks a clarifying question on ambiguous ones, without a reward model or preference pairs [1]. It is plain SFT-based data curation, not RL: contrast with RLHF-style alignment (a reward model plus a policy-gradient loop, e.g. InstructGPT [2]) and with DPO (offline preference pairs, no reward model, no sampling loop [3]) - APA uses neither preferences nor rewards, only self-computed entropy to relabel which training examples get an answer target versus a clarification target [1]. Against plain SFT on the full labeled set ("Full-set" in the paper), APA is the pick when training-set curation by the model's own uncertainty outperforms using all available labels, which the paper's Table 1 shows happens with roughly 3x to 7x fewer training samples (3,088 of 10,036 for Llama2-7B, 1,382 of 10,036 for Mistral-7B, 3,216 of 10,036 for Llama2-13B) [1].

**Variant of**: supervised fine-tuning / instruction alignment, using the same family of curated-SFT motivation as LIMA [4], which APA's own related-work and results discussion cite for the "quality over quantity" argument it builds on [1].

**Data it needs**: a QA dataset with explicit unambiguous/ambiguous labels and, for unambiguous items, a gold answer used to sort model predictions into correct/incorrect (paper uses AmbigQA for in-domain train/validation, with 5,287 unambiguous and 4,749 ambiguous training queries) [1]. No preference pairs and no reward model are required; the only extra label needed for ambiguous items is a clarification-request text, which the pipeline itself generates or samples from a fixed list [1]. It is not on-policy in the RL sense (no reward-driven sampling loop); the "perceived ambiguity" labeling pass runs once, using the base model's own current predictions and entropy, before a single SFT training run on the resulting balanced set [1]. Training scale in the paper: 3,088 balanced training examples for Llama2-7B, 1,382 for Mistral-7B, and 3,216 for Llama2-13B, versus 10,036 for the Full-set baseline on every backbone [1].

**Extra models**: none. No value network, no reference model, no reward model, and no judge model are part of the method's definition; the same base LLM is used to generate initial predictions, self-disambiguate, and (for the Generated-response variant) draft its own clarification requests, then is fine-tuned with QLoRA [1]. GPT-4o and human annotators appear only in the paper's construction of the three out-of-distribution evaluation datasets (AmbigTriviaQA, AmbigWebQuestions, AmbigFreebaseQA), not in the alignment pipeline itself [1]. Cost detail below.

**Shipped by**: no training-framework library implements APA. The paper's own code, heyjoonkim/APA on GitHub, is a script-driven research repository (`scripts/main.sh` calling `stage_0.sh`, `train.sh`, `stage_1.sh`), not a pip-installable package with an importable trainer class [5]. Building it on top of an existing trainer would mean: a scoring/sorting pass over model predictions, a self-disambiguation-and-entropy pass to compute Infogain, a response-construction step, and then a standard SFT loss (e.g. via a HF `Trainer` or `SFTTrainer`) on the resulting mixed dataset - a data-pipeline job layered on an existing SFT trainer, not a new loss or a new sampling loop.

## How it works

The loop in one line: score the model's current predictions, use a self-disambiguation entropy-drop test to relabel the errors as "perceived ambiguous" or "perceived unambiguous," build training targets accordingly, then run one SFT pass.

**Stage 1 - Initial Prediction Assessment.** Using inference template $t(\cdot)$, the model predicts $\hat{y}_{\text{unambig}}$ for unambiguous queries and $\hat{y}_{\text{ambig}}$ for ambiguous queries; each is compared against its gold label ($y$ or $y_{\text{clarify}}$) into five outcome categories shown in the paper's Fig. 3 - correct clarification (category 1), incorrect clarification response to an ambiguous query (2), correct answer (3), incorrect answer (4), and an unwanted clarification request on an unambiguous query (5) [1]. Categories 1 and 3 form $D_{\text{correct}}$; categories 2, 4, and 5 form $D_{\text{incorrect}}$ [1].

**Stage 2 - Perceived Ambiguity Detection.** For each $x \in D_{\text{incorrect}}$, the model is prompted to self-disambiguate $x$ into $\hat{x}_{\text{disambig}}$ under greedy decoding, and the pipeline computes the average per-token entropy of each string over the model's own output distribution [1]:

$$ \mathcal{H}_x = \frac{1}{N}\sum_i \mathcal{H}_{x,i}, \qquad \mathcal{H}_{x,i} = -\sum_{v \in \mathcal{V}} p_{x,i}(v)\log p_{x,i}(v) $$

and defines Infogain as the entropy drop from disambiguating:

$$ \text{Infogain}_{x,\hat{x}_{\text{disambig}}} = \mathcal{H}_x - \mathcal{H}_{\hat{x}_{\text{disambig}}} $$

Samples with Infogain above a threshold $\epsilon$ (set to 0.1 in the paper) are relabeled perceived-ambiguous, $x_{\text{ambig}}$, regardless of their ground-truth ambiguity label [1]. Toy illustration of the formula only (not paper numbers): a query with average entropy 0.693 nats (roughly a uniform 50/50 next-token distribution) that drops to 0.325 nats after self-disambiguation gives Infogain $\approx 0.368$, above the paper's 0.1 threshold, so it would be relabeled ambiguous; the direction of the test is that a genuine missing detail resolves into a confident continuation, while a query that is already clear (or one the model has no knowledge to disambiguate) shows little entropy change [1].

**Stage 3 - Response Construction.** For every $x_{\text{ambig}}$ the pipeline builds a clarification target $y_{\text{clarify}}$ two ways: Fixed, a response drawn at random from three predefined clarification phrases; or Generated, prompting the model with $x_{\text{ambig}}$ and $\hat{x}_{\text{disambig}}$ to name the ambiguous factor and produce a clarification request specific to it [1].

**Stage 4 - SFT.** $D_{\text{ambig}} = \{(x_{\text{ambig}}^j, y_{\text{clarify}}^j)\}_{j=1}^m$ is size-balanced against $D_{\text{correct}}$ (subsample the larger set, or if $D_{\text{correct}}$ is smaller, keep only the $D_{\text{ambig}}$ samples with the largest Infogain) so $n=m$, and the model is trained with standard next-token cross-entropy on $D = D_{\text{correct}} + D_{\text{ambig}}$ [1]:

$$ \min_\theta \sum_{(x,y)\in D} \sum_{i=1}^{|y|} -\log M_\theta(y_i \mid y_{<i}, t(x)) $$

Two trained variants result depending on the Stage-3 choice: APA-Fixed and APA-Gen [1].

## Cost

**Theory, from the method's own math:**
- No extra trained model: the objective (Eq. 4) is plain SFT cross-entropy on one policy, so there is no value network, reference model, or reward model to hold or update, unlike PPO-style RLHF [1][2].
- The entropy computation (Eq. 1-2) needs the full per-token output-vocabulary distribution for every generated token of both $x$ and $\hat{x}_{\text{disambig}}$, not just the sampled token - more than a plain `generate()` call returns by default, but with no gradient and no optimizer state.
- The pipeline is front-loaded, not per-training-step: Stages 1-3 are inference-only passes run once to build the training set; Stage 4 is a single, ordinary SFT run. This differs structurally from RL methods that regenerate rollouts every optimizer step.

**In practice, per framework:**
- Authors' repository (heyjoonkim/APA) [5]: fine-tuning uses QLoRA (4-bit base weights, r=4, alpha=16) via the Hugging Face PEFT library, with AdamW, batch size 32, learning rate selected from {1e-3, 5e-4, 1e-4} and epochs from {1, 2, 3} by validation performance [1]. The paper reports the Stage-4 SFT run finishes in roughly 30 minutes on one Tesla V100 GPU for its Llama2-7B/13B and Mistral-7B runs [1]. No other framework has published a cost figure for this method.

## How to use it

- Data prep: a QA dataset must carry explicit unambiguous/ambiguous labels, gold answers for unambiguous items, and (for evaluation) a way to judge whether a generated clarification correctly targets the ambiguity; the paper builds this from AmbigQA in-domain and augments five OOD sets (SituatedQA-Geo, SituatedQA-Temp, and three GPT-4o-ambiguated sets: AmbigTriviaQA, AmbigWebQuestions, AmbigFreebaseQA) [1].
- Label/target convention: unambiguous items train toward the gold answer; perceived-ambiguous items train toward a clarification-request string, either a fixed phrase or a model-generated one naming the ambiguous factor (Stage 3) [1].
- Knobs table - only one source ships a value, since no framework implements this method; treat the paper's numbers as the sole anchor, not a framework default:

| knob | paper [1] |
| --- | --- |
| Infogain threshold $\epsilon$ | 0.1 |
| QLoRA rank $r$ / alpha | 4 / 16 |
| optimizer | AdamW |
| batch size | 32 |
| learning rate (selected by validation) | one of {1e-3, 5e-4, 1e-4} |
| training epochs (selected by validation) | one of {1, 2, 3} |
| balanced training set size (Llama2-7B / Mistral-7B / Llama2-13B) | 3,088 / 1,382 / 3,216 |

- Trade-off: raising $\epsilon$ shrinks the perceived-ambiguous set and degrades F1_a, while lowering it grows the set toward the Full-set baseline's size and its lower accuracy; the paper's own threshold sweep (Fig. 5) shows APA staying above both a random-subset and an entropy-only baseline at every $\epsilon$ tested [1]. Response-variant trade-off: APA-Fixed generally scores higher than APA-Gen, because APA-Gen trains on the harder task of generating a specific clarification, not just a generic one [1].

## While it runs

- Signals and their healthy shapes: the paper's own diagnostic is the Misaligned Clarification Request Rate (MCR), the fraction of unambiguous samples the base model answered correctly (category 3) that shift to wrongly emitting a clarification request (category 5) after training; low MCR means the alignment preserved existing capability, and the paper reports APA has the lowest MCR of all trained baselines in every dataset tested (Fig. 4) [1]. There is no other framework-native logging (`clip_ratio`-style metrics) since no framework ships this trainer.
- Published reference runs: Table 1 gives per-dataset, per-backbone F1_u/F1_a numbers for APA-Fixed, APA-Gen, and all baselines (Direct, Ambig-aware, Sample-Rep, Self-Ask, Subset-Rand, Subset-Ent, Full-set) across Llama2-7B, Mistral-7B, and Llama2-13B - the reference curve to compare a reimplementation against [1].
- Degeneracies and defaults: the paper's own data-selection ablation (Table 2) isolates what Infogain buys over ground-truth ambiguity labels: on Llama2-7B/SituatedQA-Geo, selecting the ground-truth-ambiguous samples with the largest Infogain ("Max") scores F1_a 40.96, selecting them at random ("Rand") scores 39.31, and selecting the smallest-Infogain ones ("Min") scores 34.95, while APA's own perceived-ambiguity selection (which allows ground-truth-unambiguous samples in) scores 43.10 - the 8-point Max-to-Min gap and the further gain from APA's own criterion is the paper's evidence that Infogain-based selection, not just more ambiguous-labeled data, is what helps [1]. The paper attributes Min's weak performance to training samples with low Infogain as if ambiguous, when the model perceives them as unambiguous - a training/perception mismatch [1].
- Named successors: none found in a citation search of this paper via arXiv/ACL sourcing in this project's pipeline as of 2026-08-09 (69 citations recorded, no successor method identified in that search).
- Known failure modes, from the paper's own Limitations section: the method is evaluated only on short-form QA and would need adaptation for long-form/reasoning generation; it assumes a single, context-free query and does not address ambiguity that arises from conversational context; it is evaluated only on Llama2 and Mistral at 7B/13B scale, so larger-model behavior is unverified; and the paper only tries SFT, explicitly leaving RLHF [2] and DPO [3] alignment as unexplored alternatives that "could offer distinct advantages" [1].
- What the gain is - and is not: the paper's own results show APA improves both unambiguous-answer accuracy (F1_u) and ambiguity-detection accuracy (F1_a) over Direct prompting and over SFT on the full labeled set, using substantially fewer training samples (about 32% of the Llama2 in-domain data, about 13% for Mistral) [1]. This is a data-selection and target-construction gain for when to answer versus when to ask - the paper does not claim any improvement to the underlying answer quality on queries the base model could not already answer.

## Sources

[1] Kim et al., "Aligning Language Models to Explicitly Handle Ambiguity" (proposes Alignment with Perceived Ambiguity, APA), EMNLP 2024. https://arxiv.org/abs/2404.11972 - method definition (Sec. 3), datasets and implementation details (Sec. 4.1, 4.4, Appendix A), main results (Table 1, Sec. 5), ablations (Sec. 6, Table 2, Fig. 4, Fig. 5), Limitations. The paper has three arXiv revisions (v1-v3); all claims on this card are read from v3, fetched 2026-08-09 via the arXiv HTML mirror at https://arxiv.org/html/2404.11972v3 (base tag confirms v3). The unversioned abs/html URLs above resolve to whichever revision is current when opened and are not covered by this pin if a v4 is later posted.

[2] Ouyang et al., "Training language models to follow instructions with human feedback," 2022. https://arxiv.org/abs/2203.02155 - InstructGPT / RLHF, cited by [1] as an alignment alternative not used in this paper. Fetched 2026-08-09 (title confirmed via abs page).

[3] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023. https://arxiv.org/abs/2305.18290 - DPO, cited by [1] as an alignment alternative not used in this paper. The paper has three arXiv revisions (v1-v3); only the title was checked for this card, from the unversioned abs page, fetched 2026-08-09 - not pinned to a specific revision since no version-specific content was read.

[4] Zhou et al., "LIMA: Less Is More for Alignment," NeurIPS 2024. https://arxiv.org/abs/2305.11206 - cited by [1] for the data-quality-over-quantity motivation APA's results are compared against. Fetched 2026-08-09 (title confirmed via abs page).

[5] heyjoonkim/APA GitHub repository, "Pytorch implementation of 'Aligning Language Models to Explicitly Handle Ambiguity' (EMNLP 2024)." https://github.com/heyjoonkim/APA - repository structure and README describing `scripts/main.sh`, `stage_0.sh`, `train.sh`, `stage_1.sh` and `configs/main.yaml` knobs; 5 stars, not archived, no packaging/entry-point files at repository root. Fetched 2026-08-09 from the `main` branch at commit `49448a3`, the branch's HEAD commit at fetch time (pushed 2024-09-30); `main` is a moving pointer, so later commits are not covered by this card.
