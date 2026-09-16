# Distilling Step-by-Step

Paper: https://arxiv.org/abs/2305.02301 (v2, 5 Jul 2023) [1]

Prompt a large teacher LLM once for chain-of-thought rationales alongside its labels, then train a small seq2seq student to predict both the label and the rationale as two separate targets from the same input, so the rationale never has to be generated at deployment time.

**Distilling Step-by-Step** is an offline knowledge-distillation method for training small task-specific models with fewer labeled or unlabeled examples than standard finetuning or distillation require, introduced by Hsieh et al. to reduce both the deployed model size and the training-data volume needed to match or beat a large few-shot-prompted LLM [1]. Its parent is knowledge distillation, defined by Hinton et al. as compressing the knowledge of a large model (or ensemble) into a smaller model that is much easier to deploy [2]; standard task distillation trains the small model on the teacher's predicted labels as pseudo ground-truth [1]. Distilling Step-by-Step instead prompts the teacher LLM with chain-of-thought (CoT) exemplars, the technique defined by Wei et al. as prompting with a few chain-of-thought demonstrations as exemplars so that intermediate reasoning steps emerge in the output [3], to extract a natural-language rationale together with the label for each training input, then trains the small model as a multi-task learner that predicts the label from one task-prefixed copy of the input and the rationale from another [1]. The paper gives two reasons beyond data efficiency: rationales carry richer, more detailed task knowledge than a label alone, of the kind that would otherwise require many examples for a small model to infer from raw input; and unlike feeding the rationale to the student as an extra input, training it as a second output target removes the need for the teacher LLM at test time [1].

The method is evaluated only in the paper's own experiments, using 220M/770M/11B T5 models distilled from a 540B PaLM teacher [1]; there is no evidence in the sources checked here of adoption by a named downstream system. On e-SNLI, ANLI, CQA and SVAMP, Distilling Step-by-Step beats standard finetuning with 12.5%, 75%, 25% and 20% less labeled data respectively to reach the same accuracy, and with the full label set and a 540B-PaLM teacher it raises 220M T5-Base accuracy from 88.38/43.58/62.19/62.63 (standard finetuning) to 89.51/49.58/63.29/65.50 on the four datasets in turn (Table 1) [1]. The paper's headline result is a 770M T5 model that outperforms 540B few-shot-prompted PaLM using only 80% of one dataset, where standard finetuning cannot match PaLM even with 100% of the data [1]. Lineage in one line: knowledge distillation (Hinton et al., 2015 [2]) + chain-of-thought prompting (Wei et al., 2022 [3]) -> Distilling Step-by-Step (Hsieh et al., 2023 [1]); the paper cites concurrent rationale-distillation work (Ho et al. 2022, Magister et al. 2022 [1]) and the paper's own PINTO Tuning baseline keeps the teacher LLM in the loop at test time, which Distilling Step-by-Step removes [1]. The exact-title arXiv search returned a single unambiguous match, confirmed against the official code repository's own citation block, which cites arXiv:2305.02301 for this method [4].

**When to pick it**: pick it when you can call a large teacher LLM offline (via CoT prompting) to generate rationales for your training inputs, and you want a small, cheaply deployable model that needs less labeled or unlabeled data than plain finetuning or distillation to match or beat that teacher's few-shot performance [1]. Prefer plain knowledge distillation [2] if you only have teacher-predicted labels and no rationale-generation budget. Prefer PINTO Tuning, the paper's own baseline, if you can tolerate keeping the teacher LLM in the loop at inference and want to feed its rationale as an extra input rather than train it as a second output [1]. The nearest online neighbor is STaR, which iteratively generates rationales, keeps only the ones that yield a correct answer, fine-tunes on those, and repeats the loop rather than extracting rationales once offline [5].

**Variant of**: knowledge distillation [2], using chain-of-thought prompting [3] to generate the extra supervision.

**Data it needs**: an unlabeled or labeled input set plus, for each input, a teacher-generated rationale and label obtained once via few-shot CoT prompting (each prompt is an input/rationale/label triplet exemplar) [1]. The label can be human-annotated ground truth (finetuning case) or the teacher's own predicted label (distillation case) [1]. Offline: rationales and labels are extracted from the frozen teacher once before student training begins, not resampled during training [1]. The paper's runs used e-SNLI (549,367 train), ANLI-R1 (16,946), CQA (8,766) and SVAMP (720) labeled/unlabeled examples, with SVAMP additionally augmented by up to 2,305 unlabeled ASDiv examples [1].

**Extra models**: no value network, no reference model, no reward model. A teacher LLM (540B PaLM, or 20B GPT-NeoX in an ablation) is required only once, offline, to generate rationales and labels via CoT prompting; it is not loaded during student training or at deployment [1]. The student itself is a single T5 model trained with two forward/backward passes per step (label target, rationale target) through the same weights, detailed in Cost.

**Shipped by**: no major post-training library (trl, verl) implements this method as a named trainer; trl's `GKDTrainer` is a different, unrelated on-policy distillation method [6]. The paper's own reference implementation, `google-research/distilling-step-by-step` on GitHub, builds a custom `TaskPrefixTrainer` on top of Hugging Face `transformers`' `Seq2SeqTrainer`, run via `run.py --model_type task_prefix` [7]. Building it elsewhere means adding a second seq2seq loss term to an existing seq2seq trainer, not a new sampling loop, since the method is fully offline and supervised.

## How it works

The loop, once the rationale/label pairs are extracted: for every input, build two task-prefixed copies (`[label]`, `[rationale]`), run the student on each, and backpropagate a weighted sum of the two token-level cross-entropy losses [1].

**Label loss** [1, Eq. 1]:

$$ L_{label} = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i), \hat{y}_i) $$

where $\ell$ is cross-entropy between predicted and target tokens, and $\hat{y}_i$ is either the human label $y_i$ (finetuning) or the teacher-predicted label (distillation) [1].

**Multi-task objective** [1, Eq. 3]:

$$ L = L_{label} + \lambda L_{rationale}, \qquad L_{rationale} = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i), \hat{r}_i) \; [1,\text{Eq. 4}] $$

$f(x_i)$ produces $\hat{y}_i$ when the input is prefixed with `[label]` and $\hat{r}_i$ when prefixed with `[rationale]`; the rationale target $\hat{r}_i$ is not needed at test time, which is what removes the teacher LLM from deployment [1]. The paper does not state a value for $\lambda$ in the main text or appendix.

**Ablation the paper defines and rejects**: feeding the rationale as an extra input, $f(x_i, \hat{r}_i) \to \hat{y}_i$ (Eq. 2), requires the teacher LLM to generate a rationale before the student can predict, so it cannot remove the teacher at deployment; the paper trains the multi-task version (Eq. 3) instead [1]. A second ablation, single-task training on the concatenated target $[\hat{r}_i, \hat{y}_i]$ (Eq. 5), is compared in Table 2: multi-task training beats it on all four datasets (e.g. CQA 63.29 vs 61.37), and single-task training scores below plain standard finetuning on ANLI and CQA (43.50 and 61.37, versus 43.58 and 62.19) [1].

**Reference implementation's own convention** (not stated in the paper's text): the official code's `TaskPrefixTrainer.compute_loss` runs two forward passes per step and combines them as `loss = alpha * pred_loss + (1 - alpha) * expl_loss`, with the README recommending `alpha = 0.5` — i.e., label and rationale losses weighted equally [7]. This is a different parametrization from the paper's $\lambda$; this card does not attempt to map one onto the other.

**Worked example**: the paper's own illustration is a math-word-problem input, "Jesse's room is 11 feet long and 15 feet wide. If she already has 16 square feet of carpet. How much more carpet does she need to cover the whole floor?", for which CoT prompting elicits the rationale "Area = length × width. Jesse's room has 11 × 15 square feet." leading to the final answer "(11 × 15) − 16" [1]. The official code's rationale parser extracts a teacher output's rationale as everything before a fixed answer-marker phrase ("So the answer is" / "The answer is") and the label as the text after it [7]. The student then sees two copies of the same input at training time: one prefixed `[label]` with the final-answer target, one prefixed `[rationale]` with the extracted rationale text as target.

## Cost

**Theory, from the method's own math**: the multi-task loss (Eq. 3) sums two independent per-example cross-entropy terms, one over label tokens and one over (typically much longer) rationale tokens, implying two forward/backward passes through the student per training example rather than one, and a training target that is on average longer than plain finetuning's label-only target. There is no extra model held in memory (no value network, no frozen reference model, no reward model): the teacher LLM's cost is paid once, offline, for the rationale-extraction pass, and does not recur during student training.

**In practice, per implementation**: the official `TaskPrefixTrainer.compute_loss` literally calls `model(**inputs['pred'])` and `model(**inputs['expl'])` — two full forward passes per training step through the same model, and correspondingly two backward passes whose gradients are combined via `alpha * pred_outputs.loss + (1 - alpha) * expl_outputs.loss` before a single optimizer step [7]. The paper's own appendix reports T5-Base/Large runs at batch size 64 and T5-XXL (11B) at batch size 32, both with max input length 1024, on cloud A100x16 instances, for up to 10,000 steps (T5-Base/Large) or 4,000 steps (T5-XXL) [1]. No other framework's cost behavior is documented here since none ships the method.

## How to use it

- Rationale/label extraction: curate a small set of (input, rationale, label) exemplar triplets per task, use them as a few-shot CoT prompt to the teacher LLM, and parse each teacher output into a rationale and a label; the official code splits the teacher output on a fixed answer-marker phrase ("So the answer is" / "The answer is") to separate the two [1][7].
- Student input format: prepend a `[label]` or `[rationale]` task prefix to the (otherwise identical) input text, so the same T5 model is trained to emit the label from one copy and the rationale from the other [1].
- Label source: human-annotated ground truth when available (standard-finetuning-comparable setting) or the teacher's own predicted label (standard-distillation-comparable setting); the paper reports both [1].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give the value):

| knob | paper [1] | official repo default/recommendation [7] |
| --- | --- | --- |
| task-loss weight † | $\lambda$ in Eq. 3, value not stated | `--alpha`, recommended `0.5` (equal weight) |
| learning rate | $5\times10^{-5}$ (all model sizes) | `--lr`, not fixed by the repo |
| batch size | 64 (T5-Base/Large), 32 (T5-XXL) | `--batch_size`, user-set |
| max input length | 1024 | `--max_input_length`, user-set |
| max training steps | 10,000 (T5-Base/Large), 4,000 (T5-XXL) | `--max_steps`, user-set |
| teacher LLM | 540B PaLM (main results), 20B GPT-NeoX (ablation) | `--llm`, user-set |

† Not the same quantity: the paper's $\lambda$ is an additive weight on only the rationale term, $L = L_{label} + \lambda L_{rationale}$ (Eq. 3) [1]; the repo's `alpha` is a convex-combination weight over both terms, `loss = alpha * pred_loss + (1 - alpha) * expl_loss` [7]. The two columns are not settings of one knob and this card does not convert between them.

- Trade-off a run designer faces: unlabeled-data augmentation helps when the target dataset is small — the paper augments SVAMP's 720 training examples with up to the full 2,305-example ASDiv set, which improves both Distilling Step-by-Step and standard task distillation, though standard distillation still trails Few-shot CoT even after augmentation [1]. Teacher size is a second knob: Table 1 shows a 20B GPT-NeoX teacher still lifts a 220M T5 student over standard finetuning (89.12 vs 88.38 on e-SNLI), but a 540B PaLM teacher lifts it further (89.51) [1].

## While it runs

- Signals and their healthy shapes: the paper reports only final task accuracy per dataset/model-size/data-fraction combination (Figures 4-9, Tables 1-2); it does not define a live training-time monitoring signal specific to this method (e.g. no rationale-quality metric tracked during training) [1].
- Published reference runs: Table 1's four-dataset accuracy comparison (standard finetuning vs. 20B-teacher vs. 540B-teacher Distilling Step-by-Step) is the paper's own reference point for whether the method is working: 88.38/43.58/62.19/62.63 (standard finetuning) vs. 89.51/49.58/63.29/65.50 (540B-teacher Distilling Step-by-Step) on e-SNLI/ANLI/CQA/SVAMP [1].
- Degeneracies and defaults: the paper's single-task ablation (Eq. 5, concatenating rationale and label into one target) is a documented failure mode, not a hypothetical one — Table 2 shows it scoring below standard finetuning on ANLI (43.50 vs 43.58) and CQA (61.37 vs 62.19), i.e. adding rationales can hurt if they are not trained as a separate task [1]. The paper attributes SVAMP's underperformance relative to Few-shot CoT to the dataset's small size (720 examples) rather than to the method itself, and shows the gap closes with unlabeled-data augmentation [1].
- Named successors: none identified in the sources checked here (the arxiv abstract page and full text of [1], and the official repository README [7]); this is a "not searched beyond these two sources" answer, not a claim that none exist.
- Known failure modes: the paper's own limitation, stated in its ablation study, is that treating rationale and label prediction as a single joint task (rather than the paper's multi-task split) can perform worse than standard finetuning on some datasets (ANLI, CQA), consistent with prior findings that free-text rationales can harm label prediction if not handled carefully [1]. No GitHub issues on the official repository were queried for this card.
- What the gain is - and is not: the paper's own analysis is entirely about data- and model-size efficiency for matching or beating a specific teacher LLM's few-shot performance on the four evaluated datasets; it does not claim the method adds reasoning capability the teacher LLM itself lacks, since the student is trained to imitate the teacher's own rationales and labels [1].

## Sources

[1] Hsieh et al., "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes", 2023. https://arxiv.org/abs/2305.02301 - defines the method: multi-task objective, rationale extraction, datasets, Table 1/2 results, ablations, appendix hyperparameters. Fetched 2026-09-03; the PDF header reads arXiv:2305.02301v2 [cs.CL], 5 Jul 2023 - all claims above are read against v2, and the abs page is a live pointer that may later serve a newer version.

[2] Hinton, Vinyals, and Dean, "Distilling the Knowledge in a Neural Network", 2015. https://arxiv.org/abs/1503.02531 - knowledge distillation, the parent method: compressing an ensemble/large model into a single smaller, easier-to-deploy model. Fetched 2026-09-03 (abstract).

[3] Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", 2022. https://arxiv.org/abs/2201.11903 - defines chain-of-thought prompting, the elicitation technique used for rationale extraction. Fetched 2026-09-03 (abstract).

[4] google-research/distilling-step-by-step, official code repository. https://github.com/google-research/distilling-step-by-step - README's citation block confirms arXiv:2305.02301 as the paper this repository implements. Fetched 2026-09-03 (raw README.md).

[5] Zelikman, Wu, and Goodman, "STaR: Bootstrapping Reasoning With Reasoning", 2022. https://arxiv.org/abs/2203.14465 - the nearest online neighbor: iteratively generates rationales, keeps only those yielding correct answers, fine-tunes, and repeats. Fetched 2026-09-03 (abstract).

[6] Hugging Face trl, GKDTrainer documentation. https://huggingface.co/docs/trl/main/en/gkd_trainer - confirms GKD is a distinct method (Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"), unrelated to Distilling Step-by-Step. Fetched 2026-09-03. Unpinned `main`-branch docs.

[7] google-research/distilling-step-by-step, source files `README.md`, `train_utils.py`, `model_utils.py`. https://github.com/google-research/distilling-step-by-step - `TaskPrefixTrainer.compute_loss` (two forward passes, `alpha`-weighted loss), CLI args (`--alpha`, `--lr`, `--batch_size`, `--max_input_length`, `--max_steps`), rationale-parsing logic. Fetched 2026-09-03 (raw files, `main` branch, unpinned).
