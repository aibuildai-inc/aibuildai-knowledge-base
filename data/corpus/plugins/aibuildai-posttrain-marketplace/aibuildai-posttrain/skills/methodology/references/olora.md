# O-LoRA

Paper: https://arxiv.org/abs/2310.14152 (Wang et al., "Orthogonal Subspace Learning for Language Model Continual Learning", 2023) [1]. Reference code: https://github.com/cmnfriend/O-LoRA [5].

Keep LoRA's frozen base weights but train a fresh, small LoRA adapter per task while forcing it orthogonal to every previous task's adapter, so new tasks don't overwrite old ones and no past-task data has to be stored.

**O-LoRA** (orthogonal low-rank adaptation) is a continual-learning method for sequentially fine-tuning a language model on a stream of tasks, introduced in "Orthogonal Subspace Learning for Language Model Continual Learning" as a way to mitigate catastrophic forgetting while learning new tasks [1]. Its parent is LoRA, which freezes the pretrained weights and injects trainable low-rank decomposition matrices into each layer [2]. O-LoRA gives each task its own LoRA pair $\{A_t, B_t\}$, freezes every earlier task's pair once trained, and adds a loss term that pushes the new task's column space to be orthogonal to the column spaces spanned by all earlier tasks' $A$ matrices, using that subspace as a proxy for the task's gradient direction [1]. The paper gives three reasons for the design: rehearsal-based continual-learning methods must store and replay past-task data, which raises privacy concerns, while O-LoRA needs none [1]; full-model regularization methods such as EWC update every parameter for every task, whereas O-LoRA only adds marginal per-task parameters [1]; and classification-focused continual-learning methods generalize poorly to unseen tasks, whereas O-LoRA is trained with instruction tuning, which the paper credits for preserving generalization [1].

No production system adopting O-LoRA was found in the sources checked here (the arXiv paper, the authors' GitHub repository, and PEFT's documentation) — this is a claim about what this card's search covered, not a claim that no such system exists. On the paper's own standard 5-task continual-learning benchmark (T5-large, averaged over 3 task orders), O-LoRA reaches 75.8 average accuracy against 72.7 for the prior best baseline LFPT5, and 69.6 against 69.2 on a harder 15-task benchmark (Table 2) [1]; the paper's own text separately describes the standard-benchmark gain as "over 24%," a figure this card could not reconcile with the ~4% relative difference the same table shows (75.8 vs. 72.7), so both are reported here rather than resolved [1]. On a zero-shot MMLU generalization check with LLaMA-7B, the O-LoRA-constrained model scores 33.6% versus 23.3%/28.6% for otherwise-identical models trained without the orthogonality constraint (Table 3) [1]. Lineage in one line: LoRA (Hu et al., 2021 [2]) -> O-LoRA (Wang et al., 2023 [1]); no confirmed named successor was found in the sources checked. The paper title was confirmed by an exact-title arXiv search that surfaced 2310.14152 as the only match for "Orthogonal Subspace Learning for Language Model Continual Learning"; a second, unrelated paper also named "OLoRA" (orthonormal initialization for LoRA via QR decomposition) was found under the same short name and is a different method entirely — see When to pick it.

**When to pick it**: sequential instruction-tuning across a stream of tasks where past-task data cannot be stored or replayed (privacy, storage) and catastrophic forgetting must be controlled without training a full new model per task [1]. Contrast with the parent, LoRA [2]: reusing one LoRA adapter across all tasks (the paper's "SeqLoRA" baseline) forgets fast, averaging only 43.7 on the standard benchmark versus O-LoRA's 75.8 [1]. Contrast with the nearest rehearsal-based alternative: "Replay" fine-tunes the whole model and stores/replays a memory buffer of past-task samples, trading storage and privacy cost for retention, and still trails O-LoRA in the paper's own table (57.8 average) [1]. Contrast with the nearest regularization-based alternative: the paper's own EWC baseline [3] fine-tunes every model parameter, adding a penalty that discourages moving parameters the model relied on for earlier tasks [1], and it underperforms O-LoRA in the same table (50.3 average) [1]. Do not confuse this method with the same-named "OLoRA" (orthonormal low-rank adaptation), which instead initializes ordinary single-task LoRA matrices via QR decomposition to speed up convergence and is unrelated to continual learning [4].

**Variant of**: LoRA [2].

**Data it needs**: a sequence of task datasets, each of ordinary instruction-formatted $(x,y)$ text pairs — the paper's standard benchmark uses five text-classification datasets reformatted as instructions, and its long-sequence benchmark combines 15 datasets drawn from that set plus GLUE and SuperGLUE [1]. No replay buffer or stored past-task data is needed; task order is itself an experimental variable, and the paper reports results over 3 different orderings for each benchmark [1]. This is offline sequential supervised fine-tuning task-by-task (ordinary cross-entropy on target text), not on-policy sampling — there is no reward function, reward model, or policy rollout anywhere in the method.

**Extra models**: none — no value network, reward model, or reference model. The method's own definition does require holding every earlier task's frozen $\{A_i, B_i\}$ LoRA pair (used to compute the orthogonality penalty against the new task's adapter, then merged into the base weights once all tasks are done) [1]; this is accumulating frozen adapter parameters, not extra full models. Growth detail in Cost.

**Shipped by**: no established training library was found to implement O-LoRA (continual learning) as a trainer. The only implementation found is the authors' own reference code, `cmnfriend/O-LoRA` on GitHub (212 stars at the time checked), a set of training scripts for T5-large and LLaMA built on a vendored copy of PEFT, not a package with an importable trainer entry point [1][5]. Hugging Face PEFT does export `LoraConfig(init_lora_weights="olora")`, but that flag implements the unrelated, same-named orthonormal-initialization method, not this one [4][6]. Building this method on a framework means writing a custom continual-learning loop that keeps one LoRA adapter per task, freezes each completed task's adapter, and adds the orthogonality penalty to the loss — a loss-and-adapter-lifecycle addition on top of an existing LoRA trainer, not a new sampling loop.

## How it works

The loop, per task $t$ in the stream: initialize a fresh LoRA pair $\{A_t, B_t\}$, freeze every previously-trained pair $\{A_i, B_i \mid i<t\}$, train on task $t$'s data with the ordinary supervised loss plus a penalty that pushes $A_t$ toward orthogonality with every earlier $A_i$, then move to task $t+1$; once all tasks are trained, every task's LoRA update is merged into the base weights [1].

For task $t$, the update subspace is approximated as the column span of $A_t = [a_t^1, a_t^2, \dots, a_t^r]$ [1]:

$$ \mathcal{U}_t = \operatorname{span}\{a_t^1, a_t^2, \dots, a_t^r\} $$

Two subspaces $\mathcal{U}$ and $\mathcal{W}$ are orthogonal when every pair of their basis vectors has zero inner product; for the LoRA subspaces of tasks $i$ and $t$ this condition is written as [1]:

$$ O_{i,t} = A_i^{T} A_t = 0 $$

The full training objective for task $t$ combines the ordinary log-likelihood on that task's data with an orthogonality penalty against every earlier task, weighted by $\lambda_1$ [1]:

$$ \sum_{x,y \in \mathcal{D}_t} \log p_{\Theta}(y \mid x) + \lambda_1 \sum_{i=1}^{t-1} L_{orth}(A_i, A_t) $$

where the orthogonality loss is the squared Frobenius norm of $O_{i,t}$, i.e. the sum of the squared entries of $A_i^{T}A_t$ [1]:

$$ L_{orth}(A_i, A_t) = \sum_{j,k} \| O_{i,t}[j,k] \|^2 $$

LoRA is applied only to the attention query and value projections $W_q, W_v$, following the parent paper [1][2]. Once all tasks are trained, every task's low-rank update is folded into the base weight matrix so inference cost does not grow with the number of tasks [1]:

$$ W_{init} := W_{init} + \sum_{i=1}^{t} A_i B_i $$

Worked micro-example (purely illustrating the formula, not tied to any implementation's tensor layout): with rank $r=1$ and $d=2$, suppose task 1's adapter column is $A_1 = [1, 0]^T$ and task 2's adapter column starts training at $A_2 = [0.6, 0.8]^T$. Then $O_{1,2} = A_1^T A_2 = 0.6$, so $L_{orth}=0.6^2=0.36$; the gradient of this penalty pushes $A_2$ toward the direction where the dot product with $A_1$ is zero, e.g. toward $[0,1]^T$, at which point $L_{orth}=0$ and task 2's update no longer overlaps task 1's subspace.

The paper's appendix separately lists per-task values of a second coefficient, $\lambda_2$, used across its long-sequence-benchmark task orders, but the objective given in the paper's own Eq. 7 above contains only $\lambda_1$; the text nowhere else defines what $\lambda_2$ multiplies [1]. This is not this card's misreading: an open, unanswered issue on the authors' repository asks the same question — "the main paper explains λ1, but has no mention of λ2... Could you please explain the λ2 parameter?" — with no reply as of the check date [7]. Reading the reference implementation's loss computation (commit `8e91796`, `src/uie_trainer_lora.py`) resolves it and reveals two further paper-vs-code gaps: the code's orthogonal loss is `torch.abs(torch.mm(param, param_.T)).sum()` — a sum of absolute values over the frozen-vs-new $A$ matrices — not the sum-of-squares that Eq. 8 states, and the code adds a second term, `l2_loss`, that is the L2 norm of only the *new* task's LoRA $A$ and $B$ matrices, weighted by $\lambda_2$, which does not appear anywhere in the paper's stated objective [8]. So the shipped reference loss is `accuracy_loss + λ1 * orthogonal_loss(abs-sum) + λ2 * l2_loss(new-adapter-norm)`, not Eq. 7 as written [8].

## Cost

**Theory, from the method's own math**: the extra compute per task is one matrix product $A_i^T A_t$ (a $d\times r$ by $d \times r$ product reduced to an $r \times r$ matrix) for every earlier task $i<t$, plus summing its squared or absolute entries — negligible next to a forward/backward pass through the backbone; the paper states that because only the training loss is modified, "no additional training cost" is incurred [1]. Memory grows by one frozen LoRA pair $(A_i \in \mathbb{R}^{d\times r}, B_i \in \mathbb{R}^{r\times k})$ per completed task, so total held adapter parameters scale as $O(T \cdot r \cdot (d+k))$ for $T$ tasks — tiny per task relative to the backbone, but not literally free forever: every earlier task's $A_i$ must stay resident (or be reloaded) to compute the orthogonality penalty against each new task, and all of them must be held until the final merge (Eq. 9) [1]. No value network, reward model, or reference model is trained or held at any point, unlike RL post-training methods.

**In practice**: no established training framework was found to ship this method (see Shipped by), so the only practice-level cost data available is the authors' own reported run: T5-large experiments used 8 NVIDIA RTX 3090 GPUs with DeepSpeed, one epoch per task, a batch size of 64 (8 per GPU) [1, Appendix A.1]. This is one paper's own hardware/config report, not a framework-level cost claim, and should not be generalized to other backbones or scales.

## How to use it

- Data prep: format each task's examples as ordinary instruction-tuned $(x,y)$ text pairs; task order is a real experimental variable — the paper evaluates 3 orders on its 5-task benchmark and 3 more on its 15-task benchmark, and reports averages over orders rather than a single fixed order [1].
- Reward/label conventions: none. Training is standard supervised cross-entropy on target text; there is no reward function, preference label, or judge anywhere in the method.
- Because no framework ships this method, there is no framework-default column to compare against; the table below gives only the paper's own values, read from its main text and Appendix A.1 [1].

| knob | paper's value [1] |
| --- | --- |
| LoRA target modules | $W_q$, $W_v$ only |
| rank $r$ | ablated at 2, 4, 8, 16 (T5-Base); 2 vs. 16 not significantly different |
| $\lambda_1$ (orthogonality weight) | 0.5 for most tasks/orders; up to 5 for some tasks in the 15-task orders |
| $\lambda_2$ | per-task values given in Appendix A.1; the term it multiplies is undocumented in the paper's own objective (see How it works) |
| learning rate | 1e-3, constant |
| batch size | 64 (8 per GPU x 8 GPUs) |
| epochs per task | 1 |
| dropout | 0.1 |
| weight decay | 0 |

- Rank trade-off: a larger $r$ gives each task a bigger subspace to occupy and slightly more room for the orthogonality constraint to bind without hurting current-task fit, but the paper's own ablation found the average accuracy gain from $r=2$ to $r=16$ small, and interprets this as evidence that the model's gradient space has low intrinsic dimensionality [1].
- $\lambda_1$ trade-off: the paper's own ablation compares $\lambda_1=0.5$ against $\lambda_1=0$ (i.e., a new LoRA adapter added per task with no orthogonality constraint) and shows the constrained version keeps the prediction loss on old-task samples lower after training a new task, which is the paper's evidence that the orthogonality term is what protects old tasks rather than merely adding capacity [1].

## While it runs

- Signals and their healthy shapes: the paper's own diagnostic is not a live training-curve metric but a post-hoc check — comparing the change in prediction loss on old-task samples after training a new task, with $\lambda_1=0.5$ against $\lambda_1=0$, as evidence the constraint is working [1]; and a hidden-state-variation check across T5 layers, showing lower layers (generic semantic knowledge) change little across tasks while higher layers (task-specific knowledge) change more, and that the orthogonality constraint reduces that variation [1]. No library-level per-step logging convention (comparable to trl's or verl's crucial-values lists) was found for this method, because no library ships it.
- Published reference runs: the paper's own Table 2 (standard 5-task and long 15-task benchmarks, 3 orders each, T5-large) and Table 3 (MMLU zero-shot generalization with LLaMA-7B) are the reference numbers to compare against; both are reported above in Adoption and Results [1].
- The deciding number: comparing the full method against the same paper's IncLoRA baseline — "incremental learning of new LoRA parameters on a sequential series of tasks (without adding any regularization or replaying samples from the previous tasks)" [1], which is exactly O-LoRA with $\lambda_1=0$ — shows the orthogonality term is what makes the method work: IncLoRA averages 66.4 on the standard benchmark and 61.2 on the long-sequence benchmark, versus O-LoRA's 75.8 and 69.6 (Table 2) [1].
- Degeneracies and defaults: the paper's own Limitations section states the method still requires task identification during training (to route each example to the correct per-task LoRA), even though it does not need task IDs at inference, and that behavior at scales of hundreds of tasks is untested [1]. Separately, an unanswered issue on the reference implementation reports a live run where the logged `orthogonal_loss` stays at exactly 0.0 across steps despite $\lambda_1=0.5$ being active, alongside a nonzero `l2_loss` — a reported, unconfirmed-by-maintainer degeneracy in that specific codebase, not a property derived from the paper's math [9].
- Named successors: none found in the sources checked (the paper itself, its GitHub repository, and its citing issues); no broader forward-citation search for successor methods was performed for this card.
- Known failure modes: from the paper's own Limitations section — performance at "hundreds of tasks" is unverified, and the method still needs task identity at training time, which the authors flag as a direction for future task-agnostic training [1]. From the reference implementation's issue tracker: three open issues (#22, #23, #32) report LLaMA reproduction accuracy falling well short of the paper's reported numbers on the same benchmark order [10][11][12]. In issue #23's own reported run, per-task LLaMA2 predict-exact-match was 97.6% on the first task but had fallen to 33.4% by the second task in the sequence, a within-run collapse the reporter attributes to catastrophic forgetting rather than to any single number in the paper [11]; the maintainer's replies to #23 and #32 attribute reproduction gaps to sensitivity to GPU count and suggest raising $\lambda_1$/$\lambda_2$ or lowering the learning rate to compensate, rather than confirming a code bug [11][12].
- What the gain is - and is not: the paper's own analysis targets forgetting and generalization to unseen tasks, not new capability — its MMLU comparison shows models trained without the orthogonality constraint drop to near-random accuracy (23.3%/28.6%, against a 25% random-guess floor on the 4-way MMLU format) while the O-LoRA-constrained model retains 33.6%, which the paper frames as preserving pre-existing generalization rather than adding new capability [1].

## Sources

[1] Wang et al., "Orthogonal Subspace Learning for Language Model Continual Learning", 2023. https://arxiv.org/abs/2310.14152 — defines O-LoRA: motivation, objective (Eqs. 3-9), baselines, Tables 2/3/5, Appendix A.1 implementation details, Limitations. Fetched 2026-08-09 (ar5iv HTML full text).

[2] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. https://arxiv.org/abs/2106.09685 — LoRA, the parent method. Fetched 2026-08-09 (abstract page).

[3] Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", 2017 — EWC, cited via [1]'s own baseline description ("finetune the whole model with a regularization loss that prevents updating parameters that could interfere with previously learned tasks"); the EWC paper itself was not separately fetched for this card.

[4] Büyükakyüz, "OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models", 2024. https://arxiv.org/abs/2406.01775 — the unrelated, same-named orthonormal-initialization method; fetched to confirm it is a different method (single-task convergence speed via QR-decomposition init, not continual learning). Fetched 2026-08-09 (ar5iv HTML full text and abstract page).

[5] cmnfriend/O-LoRA GitHub repository (README and repository metadata). https://github.com/cmnfriend/O-LoRA — the paper's own reference implementation; 212 stars at the time checked. Fetched 2026-08-09.

[6] Hugging Face PEFT, LoRA package reference (`main` branch). https://raw.githubusercontent.com/huggingface/peft/main/docs/source/package_reference/lora.md — documents `LoraConfig(init_lora_weights="olora")` as implementing [4], not [1]. Fetched 2026-08-09, unpinned `main`-branch build.

[7] cmnfriend/O-LoRA GitHub issue #40, "what is λ2 in the appendix?", opened 2025-05-19. https://github.com/cmnfriend/O-LoRA/issues/40 — open, no reply as of the check date. Fetched 2026-08-09 via GitHub API.

[8] cmnfriend/O-LoRA, `src/uie_trainer_lora.py`, commit `8e91796bcab50c29ae20172a54aed14d373d37be` (2023-10-10). https://github.com/cmnfriend/O-LoRA/blob/main/src/uie_trainer_lora.py — the reference implementation's actual loss computation (`orthogonal_loss`, `l2_loss`, their weighting by `lamda_1`/`lamda_2`). Fetched 2026-08-09; commit-pinned as stated.

[9] cmnfriend/O-LoRA GitHub issue #41, "orthogonal loss error", open, no reply as of the check date. https://github.com/cmnfriend/O-LoRA/issues/41. Fetched 2026-08-09 via GitHub API.

[10] cmnfriend/O-LoRA GitHub issue #22, "llama2的结果比论文中的llama1的结果低" ("LLaMA-2 results are lower than the paper's LLaMA-1 results"), opened 2024-05-24, three users reporting reproduction shortfalls, no maintainer resolution found as of the check date. https://github.com/cmnfriend/O-LoRA/issues/22. Fetched 2026-08-09 via GitHub API.

[11] cmnfriend/O-LoRA GitHub issue #23, "llama2 结果复现" ("reproducing LLaMA2 results"), opened 2024-06-05, per-task LLaMA2 predict-exact-match reported as low as 33.4%; maintainer reply attributes the gap to GPU-count sensitivity and suggests raising lambda1/lambda2 or lowering the learning rate. https://github.com/cmnfriend/O-LoRA/issues/23. Fetched 2026-08-09 via GitHub API.

[12] cmnfriend/O-LoRA GitHub issue #32, "Question Regarding Reproducing LLaMA Results", opened 2024-09-02, DBpedia exact-match reproduced at up to 91% against an expected ~98%; maintainer reply again attributes the gap to GPU-count sensitivity and suggests reducing the learning rate to 1e-04. https://github.com/cmnfriend/O-LoRA/issues/32. Fetched 2026-08-09 via GitHub API.
