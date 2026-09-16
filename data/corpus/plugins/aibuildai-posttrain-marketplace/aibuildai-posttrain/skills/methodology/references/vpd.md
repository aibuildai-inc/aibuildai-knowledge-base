# VPD

https://arxiv.org/abs/2312.03052

Sample several candidate visual programs per query with an LLM, keep the one whose execution matches the label, rewrite its trace into a chain-of-thought, and distill both the answer and the chain-of-thought into a single vision-language model with two cross-entropy losses.

**VPD** (Visual Program Distillation) is an instruction-tuning framework for producing a vision-language model (VLM) that solves complex visual tasks in a single forward pass, introduced by Hu et al. [1]. Its parent is ViperGPT, a framework that composes vision-and-language sub-models by having an LLM write a Python program against a provided vision-tool API and then executing that program, requiring no training of its own [2]. VPD keeps ViperGPT's program-generation-and-execution machinery but uses it only to synthesize training data: it samples several candidate programs per query, executes each, keeps the one whose output matches the ground-truth label, rewrites the kept program's execution trace into a natural-language chain-of-thought (CoT) with an LLM, and fine-tunes a backbone VLM to predict both the answer and the CoT [1]. The paper gives two reasons for this design: LLM-generated visual programs are error-prone — they omit or add steps and cannot recover when a sub-model returns a wrong output — and running the multi-model program pipeline at inference time is slow and expensive, since it loads several specialized vision models per query [1]. Searching by the acronym "VPD" collides with an unrelated 2025 reinforcement-learning paper, "Learning from Language Feedback via Variational Policy Distillation" (arXiv:2605.15113, 2 citations) [3]; the paper cited above is the intended match, confirmed by a top-cited pick between the two same-acronym candidates (90 citations versus 2).

The only systems reported using VPD are the paper's own two backbones, PaLI-3-VPD (5B) and PaLI-X-VPD (55B) [1]; no external landmark system was found citing VPD as an adopted training method (see While it runs for the citation search). In the originating paper, PaLI-X-VPD sets a new state of the art among generalist VLMs, beating the prior best generalist VLM by +8.5 on MMBench and +9.8 on TallyQA-complex, and beating the specialist state of the art by +9.5 on A-OKVQA† [1], and beats its own no-CoT instruction-tuned twin, PaLI-X-Instruct, on 11 of 12 evaluated tasks, including +1.6 on GQA, +1.2 on TallyQA-complex, and +1.2 on MMBench (Table 1) [1]. Lineage in one line: ViperGPT (2023 [2]) -> VPD (2023 [1]) -> distilled backbones PaLI-3-VPD / PaLI-X-VPD (the paper's own models [1]); no named successor was found (see While it runs).

† The MMBench and TallyQA-complex deltas are against the prior best generalist VLM, while the A-OKVQA delta is against the specialist state of the art (InstructBLIP); the paper itself marks only the A-OKVQA figure with this qualifier, so the three numbers are not directly comparable to one another [1].

**When to pick it**: pick VPD when the target task decomposes into vision-tool calls and short-answer labels are available (or can be trusted) to verify LLM-generated programs against, and a single fast forward pass at inference matters. Compared with its parent, ViperGPT, which needs no training but re-runs the LLM-plus-tool-API pipeline at every inference call [2], VPD pays that cost once during data synthesis and ships a single model. Compared with plain instruction tuning on the same (image, query, label) triples without rationale supervision — the paper's own PaLI-Instruct baseline — the added CoT loss is what produces VPD's accuracy and faithfulness gains in Table 1 and the human evaluation [1]. Compared with Distilling Step-by-Step, whose two-loss label-plus-rationale training recipe VPD reuses [4], VPD is vision-specific: its rationales come from verified, executable programs over a vision-tool API rather than from an LLM's own free-form explanation.

**Variant of**: ViperGPT's visual-program framework [2], trained with the label-plus-rationale dual-loss recipe of Distilling Step-by-Step [4].

**Data it needs**: (image, textual query, short expected-answer label) triples from academic VQA-style datasets. VPD's own generalist training draws on VQAv2, OCR-VQA, GQA, OK-VQA, A-OKVQA, and TallyQA, totaling 310.5K labeled examples, of which 89.6K received a synthesized CoT after program-verification filtering (GQA contributed 38.0K CoTs from 86.0K labels, OK-VQA 6.7K from 9.0K, A-OKVQA 11.2K from 17.1K, TallyQA 33.7K from 48.4K) [1]. Offline: the CoT-augmented dataset is synthesized once, before VLM fine-tuning begins, by the LLM-plus-vision-tool pipeline; the fine-tuning stage itself trains on this fixed dataset rather than on fresh policy samples [1].

**Extra models**: none during the VLM's own fine-tuning or at inference — a single backbone VLM (PaLI-3 or PaLI-X) is trained with LoRA on both losses, and the trained model answers alone at inference [1]. Extra models exist only in the one-time data-synthesis pipeline and are discarded afterward: PaLM-2 for program generation and for rewriting execution traces into CoTs, plus the vision-tool API's own sub-models — PaLI-X-based detection distilled from OWLv2, the Google Cloud Depth API, and PaLM-2 again for external-knowledge queries [1]. See Cost for the resulting inference-latency payoff.

**Shipped by**: no library implements VPD. A GitHub repository search for "visual program distillation" returns only the paper's own project-website repository, which carries no training code and has 0 stars [5][6]. Building it requires two different jobs: a ViperGPT-style program-generation-and-execution harness — an LLM plus a Python vision-tool API — to synthesize the CoT data [2], and a standard SFT loop that adds a second cross-entropy loss predicting the CoT alongside the label loss on top of an existing VLM trainer [1].

## How it works

For each labeled (image, query, answer) example: sample k candidate Python programs from an LLM, execute each against a fixed vision-tool API, keep the one whose output matches the label, rewrite its execution trace into a natural-language chain-of-thought, then fine-tune the backbone VLM to predict both the short answer and the chain-of-thought [1].

**Program generation and filtering.** PaLM-2 is prompted with a description of the available vision modules and the query, and is expected to output a Python function; because top-1 success is much lower than top-k success, the paper samples $k=5$ candidate programs at temperature $T=0.5$ when a ground-truth label is available, and $k=1$ (no filtering) when it is not [1]. Any program that fails to execute is discarded; among the programs that execute, the one whose output matches the human label is kept, with an LLM used to judge correctness for ambiguously-phrased answers, and the top-scoring program (by the generating LLM's own score) kept if more than one is correct [1]. If no sampled program is verified correct, the example is not discarded: it still trains the label loss alone [1].

**Distillation objective** [1]:

$$ \mathcal{L} = \mathcal{L}_{label} + \mathcal{L}_{rationale} = \sum_{j=1}^{N} \ell\big(f(i_j, q_j, p_j),\, \hat{y}_j\big) + \ell\big(f(i_j, q_j, s_c),\, c_j\big) $$

Here $f$ is the VLM; $i_j$ is the image and $q_j$ the textual query of example $j$; $p_j$ is the task-specific instruction prompt (e.g. "Answer with a single word or phrase"); $\hat{y}_j$ is the kept program's output, which equals the ground-truth label $y_j$ whenever the filtering step succeeded; $s_c$ is a fixed suffix, "Explain the rationale to answer the question", used to elicit the long-form answer; $c_j$ is the CoT rationale rewritten by an LLM from the kept program's execution trace; and $\ell$ is cross-entropy loss normalized by sequence length, applied separately to the (typically short) label and the (typically longer) rationale so that neither term dominates by length alone [1]. The two forward passes for one example share the same image and query but different prompt suffixes ($p_j$ vs. $s_c$) and different targets ($\hat{y}_j$ vs. $c_j$); their losses are added with no separate weighting [1].

**Worked example of the filtering payoff.** Sampling $k=5$ programs instead of $k=1$ raises the fraction of queries for which at least one sampled program passes filtering by +45 percentage points on GQA and A-OKVQA, +33 on OK-VQA, and +10 on TallyQA [1] — this is what the paper credits for most of the added CoT-training data.

**Process-supervision variant.** The paper does not define a separate advantage or credit-assignment rule beyond the two losses above; program filtering and CoT rewriting happen once per example before training, not per token or per step, so there is no analogue of a per-token advantage in VPD.

## Cost

**Theory, from the method's own math:**

- Time: two phases with different cost profiles. Data synthesis is paid once, before any VLM training step: for each training example it samples up to $k=5$ programs from an LLM, executes each against the vision-tool API (which itself calls several sub-models), and rewrites the kept trace into a CoT with another LLM call [1] — this cost is amortized over however many training epochs the synthesized dataset is reused for. VLM fine-tuning itself needs two forward-backward passes per example, one for $\mathcal{L}_{label}$ and one for $\mathcal{L}_{rationale}$, roughly doubling the per-example compute of single-loss instruction tuning on the same labels [1].
- Memory: only one model is trained — a single backbone VLM fine-tuned with LoRA — so no critic, reward model, or reference model is held during this stage [1]. The teacher pipeline's LLM and vision-tool sub-models are needed only during the one-time data-synthesis phase, not while the backbone VLM is being fine-tuned or at inference [1].
- The naive reading to flag: VPD's two-loss objective can look like standard knowledge distillation, which is sometimes implemented by holding a live teacher model alongside the student during training. VPD does not do this — the teacher's outputs (verified answers and CoTs) are baked into a static dataset before backbone fine-tuning starts, so no teacher model needs to be resident in memory during that stage [1].

**In practice, per framework:**

- No framework or library ships VPD, so the only reported run is the paper's own: PaLI-X-VPD (55B) trained on 128 TPU-v3 for about 2 days, PaLI-3-VPD (5B) on 32 TPU-v4 for about 20 hours, both with LoRA rank 8 on every linear layer of the encoder and decoder attention and MLP blocks, batch size 128, 8,000 steps, cosine learning-rate schedule with 1% warmup and peak learning rate 1e-4, and the AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.98$) in bfloat16 [1][7].
- The paper's inference-cost measurement, over 300 questions on 128 TPU v5: the ViperGPT-style teacher pipeline takes 4.7s for code generation plus 4.2s for program execution, 8.9s per question total, versus 0.8s per question for the distilled PaLI-X-VPD model [1] — an approximately 11x latency reduction, which is VPD's central practical payoff.

## How to use it

- Data prep: gather (image, query, short-answer label) triples from existing VQA-style datasets, and build a Python vision-tool API (object/knowledge/depth queries etc.) that the program-generation LLM can call, following ViperGPT's execution-engine design [2]; the LLM is then prompted with the API description and the query to write a program [1].
- Label/verification convention: a sampled program is kept when its executed output exactly matches the ground-truth label, with an LLM used as a judge for ambiguous phrasing [1]; when no sampled program passes, the example is kept for the label loss only, with $\mathcal{L}_{rationale}$ set to 0 rather than the example being dropped [1]; for unlabeled data the filtering step is skipped entirely and the top-scoring executable program is used directly [1].
- Key knobs, with the paper's own values as anchors (no framework ships defaults to compare against; "n/a" marks a knob that does not apply to that stage):

| knob | VPD generalist training [1] | VPD per-task fine-tuning [1] |
| --- | --- | --- |
| LoRA rank | 8 (encoder + decoder, all linear layers) | 4 (PaLI-X-VPD) / 8 (PaLI-3-VPD), encoder only |
| peak learning rate | 1e-4, cosine schedule, 1% warmup | 1e-4, cosine schedule, 1% warmup |
| batch size | 128 | 64 |
| steps / epochs | 8,000 steps | 1 epoch (GQA), 3 epochs (other tasks) |
| optimizer | AdamW, $\beta_1=0.9$, $\beta_2=0.98$, bfloat16 | same |
| candidate programs $k$ | 5 (labeled data), 1 (unlabeled data) | n/a |
| sampling temperature | $T=0.5$ | n/a |

- Trade-off: raising $k$ (candidate programs sampled per query) sharply raises the fraction of examples that get a verified CoT — +45pp on GQA and A-OKVQA, +33pp on OK-VQA, +10pp on TallyQA going from $k=1$ to $k=5$ [1] — at the cost of $k\times$ the LLM-sampling and program-execution calls during synthesis. There is no batch-reuse trade-off analogous to an RL rollout batch: VPD fine-tunes in a single pass over a fixed, pre-synthesized dataset, not by resampling from a live policy [1].

## While it runs

- Signals and their healthy shapes: the paper reports no named logged metrics or live curves (there is no framework or logging guide for VPD); the one first-hand signal it gives is that training still showed a steady loss drop when terminated at 8,000 steps, which the authors take to mean more compute would likely help further [1].
- Published reference runs: Table 1 gives PaLI-X-VPD vs. PaLI-X-Instruct at matched hyperparameters — GQA 64.9 vs. 63.3 (+1.6), TallyQA-complex 76.6 vs. 75.4 (+1.2), MMBench 76.2 vs. 75.0 (+1.2) — and vs. prior generalist SOTA, +8.5 on MMBench and +9.8 on TallyQA-complex, plus +9.5 on A-OKVQA against the specialist SOTA† [1]. The paper's human evaluation (600 samples from GQA, A-OKVQA and others) found PaLI-X-VPD +16.7 percentage points more accurate than PaLI-X-Instruct, able to explain its answer on +24% more samples, with rationales judged factual 87.2% of the time (+14.6pp) and consistent 97.8% of the time (+10pp), and preferred by annotators on +25% more samples overall and +12% more samples when both models answered correctly [1].
- Degeneracies and defaults: if the LLM sampling finds zero programs whose output matches the label, VPD does not drop the example — it sets $\mathcal{L}_{rationale}=0$ and trains on $\mathcal{L}_{label}$ alone [1]. For unlabeled data the verification step is skipped and the top-scoring executable program is used without any check against ground truth [1]. No configuration defaults diverge between "paper" and "framework" because no framework ships VPD.
- Named successors: none found. Semantic Scholar's citing-papers list for arXiv:2312.03052, queried on 2026-08-09 (50 entries returned), surfaced tool-augmented VLM and distillation papers — e.g. "Reinforced Visual Perception with Tools" and "SpatialCLI: Learning to Reason With Spatial Tools, Then Without Them" — but none frame themselves as fixing a documented bias of VPD the way DAPO or Dr. GRPO fix GRPO's length bias [8].
- Known failure modes: from the paper's own Limitations section — programs struggle in scenes with occluded objects or many objects [1]; a single static program cannot handle tasks that need interactive re-planning, which the authors suggest LLM-agent approaches could address [1]; and generated CoTs are not independently fact-checked beyond matching the program's output to the label, which the authors flag as a gap since even strong VLMs like PaLI-X and GPT-4V cannot reliably verify claims themselves [1]. No maintainer-issue-tracker failure modes exist because no library ships VPD (see Shipped by).
- What the gain is - and is not: the paper's own results show the gain is in answer accuracy and, more markedly, in the faithfulness, consistency, and explainability of a single-pass model's rationale, not in added world knowledge — PaLI-3-VPD (5B) does not beat prior SOTA on knowledge-based VQA tasks, which the authors attribute to its 3B UL2 language model being too small to hold the knowledge those tasks need [1]. VPD also cannot exceed what its teacher pipeline could sample in the first place: it distills only from a program that at least one of the $k$ sampled candidates got right [1].

## Sources

[1] Hu et al., "Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models", 2023. https://arxiv.org/abs/2312.03052 — defines VPD: program sampling/filtering, CoT conversion, dual-loss objective, data mixture (Table 4), training details (Appendix B.3), inference costs (Appendix B.4), quantitative results (Table 1), human evaluation (§4.3), limitations (§6). Fetched 2026-08-09 (arXiv abstract page and full HTML text).

[2] Surís, Menon, and Vondrick, "ViperGPT: Visual Inference via Python Execution for Reasoning", 2023. https://arxiv.org/abs/2303.08128 — the parent visual-program framework VPD distills from. Fetched 2026-08-09 (arXiv abstract page).

[3] "Learning from Language Feedback via Variational Policy Distillation", 2025. https://arxiv.org/abs/2605.15113 — the unrelated reinforcement-learning paper VPD's acronym collides with. Fetched 2026-08-09 (arXiv abstract page).

[4] Hsieh et al., "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes", 2023. https://arxiv.org/abs/2305.02301 — origin of the label-plus-rationale dual-loss training recipe VPD reuses, per [1]'s own citation. Fetched 2026-08-09 (arXiv API metadata; not independently read beyond title/id confirmation).

[5] GitHub code search API, query "visual program distillation", queried 2026-08-09. https://api.github.com/search/repositories?q=visual+program+distillation — used to confirm no training-code library implements VPD.

[6] VPD project website. https://visual-program-distillation.github.io/ — confirms the only public repository is the project webpage, with no training code released. Fetched 2026-08-09.

[7] Chen et al., "PaLI-3 Vision Language Models: Smaller, Faster, Stronger", 2023 (https://arxiv.org/abs/2310.09199), and Chen et al., "PaLI-X: On Scaling up a Multilingual Vision and Language Model", 2023 (https://arxiv.org/abs/2305.18565) — the two backbone VLMs VPD fine-tunes, cited by [1] for architecture and size (5B / 55B). Confirmed via arXiv API metadata, not independently read beyond title/id.

[8] Semantic Scholar Graph API, citations of arXiv:2312.03052, queried 2026-08-09. https://api.semanticscholar.org/graph/v1/paper/arXiv:2312.03052/citations — used for the named-successor search; 50 citing-paper titles returned and screened.

Framework docs are not applicable to this card: no library implements VPD, so there is no unpinned "main"/"latest" build to date-stamp beyond the arXiv paper's own PDF (v2, dated 2024-04-05) and the two web resources in [5] and [6], both read live on 2026-08-09.
