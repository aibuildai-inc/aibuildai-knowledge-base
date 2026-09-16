# LoRA

Freeze the pre-trained weight matrix and train two small low-rank matrices that add a correction on top of it, so only a tiny fraction of parameters need gradients and optimizer state.

**LoRA** (Low-Rank Adaptation of Large Language Models), the paper's page is at https://arxiv.org/abs/2106.09685, is a parameter-efficient fine-tuning method that Hu et al. define as an approach that "freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks" [1]. It is a variant of standard fine-tuning [1], which the paper contrasts with two families it improves on: adapter tuning, which inserts extra trainable layers into the network and thereby adds inference latency [2] ([1]), and prefix/prompt-tuning methods, whose limitation the paper describes as shrinking the sequence length still usable for the actual task [1]. For a pre-trained weight matrix $W_0$, LoRA represents the update as a low-rank product $W_0 + \Delta W = W_0 + BA$ with rank $r \ll \min(d,k)$, and only $A$ and $B$ receive gradients while $W_0$ stays frozen [1]. The paper gives three reasons for the design: full fine-tuning of models the size of GPT-3 175B is prohibitively expensive to store and deploy per task [1]; prior parameter-efficient methods trade away either inference latency or usable context length to get their savings [1]; and the paper's own intrinsic-rank experiments motivate the hypothesis that the weight update itself, not just the model's activations, has low intrinsic rank during adaptation [1].

LoRA is now the default parameter-efficient fine-tuning method behind Hugging Face's PEFT library, whose LoRA quick-start applies a `LoraConfig` to freeze the base model and train only the injected low-rank matrices [3]. It is the mechanism underneath QLoRA, which sends gradients back through a frozen, 4-bit quantized base model into LoRA adapters, and reports this lets a 65B-parameter model be fine-tuned on one 48GB GPU without giving up the task performance of full 16-bit fine-tuning [4]. In the originating paper's own GPT-3 175B stress test (Table 4), LoRA with 4.7M trainable parameters reached 91.7% MNLI-m accuracy and 53.8/29.8/45.9 SAMSum ROUGE-1/2/L, matching or exceeding full fine-tuning's 175,255.8M-parameter run (89.5% MNLI-m, 52.0/28.0/44.5 ROUGE) [1]. Lineage in one line: full fine-tuning / adapter tuning [2] / prefix-tuning -> LoRA (2021, [1]) -> QLoRA (2023, [4]) -> rank/scaling successors LoRA+ and rank-stabilized LoRA [5][6]. The paper was confirmed as the correct match by its dominant citation count among candidates sharing the "low-rank adaptation" name (21,482 citations versus the next closest at 453, per the shortlist's top-cited-collision-pick rule).

**When to pick it**: pick LoRA whenever full fine-tuning's memory or storage cost is the blocker and the task can tolerate a bounded, low-rank weight update - it is orthogonal to the training objective, so it composes with SFT, DPO, or an RL loop such as PPO or GRPO rather than competing with them [1][7]. Prefer full fine-tuning, LoRA's parent, when the update needed is not well approximated by a small rank and the extra capacity is worth the memory cost [1]. Prefer adapter tuning [2], the nearest earlier alternative, only if extra inference latency from added layers is acceptable, since LoRA's linear form merges into the base weights and adds none [1]. There is no "nearest online neighbor" in the online/offline sense used for RL methods: LoRA is a weight-parameterization choice, not an on-policy/offline training rule, and it plugs into either kind of trainer [7].

**Variant of**: standard (full) fine-tuning [1], contrasted with adapter tuning [2].

**Data it needs**: whatever the wrapped objective needs - LoRA supplies no data format of its own. In the paper's own experiments this ranged from GLUE-style labeled classification pairs (RoBERTa/DeBERTa) to prompt-completion pairs for GPT-2/GPT-3 (E2E NLG, WikiSQL, SAMSum, ~144K-scale instruction sets are typical in later adopters, not stated by this paper) [1]. On-policy versus offline is likewise inherited from the host trainer, not set by LoRA: verl documents using LoRA inside PPO and GRPO RL loops, which are on-policy [7], while PEFT's own quick-start example is a supervised classification fine-tune, which is offline [1][3].

**Extra models**: none required by LoRA's own definition - it adds only the two low-rank matrices $A$, $B$ per targeted layer, and needs no value network, reference model, reward model, or judge [1]. Any such model in a given run comes from the host method (e.g., GRPO's optional reference model), not from LoRA. See Cost for what stays frozen versus trained.

**Shipped by**: Hugging Face PEFT, whose top-level package exports `LoraConfig` and `get_peft_model` as the entry point [3], is the reference implementation and is under active development as HF's core parameter-efficient fine-tuning library. TRL wraps PEFT rather than reimplementing LoRA: passing a `peft_config` (or the `--use_peft` CLI flag) to `SFTTrainer` and other trainers attaches PEFT's `LoraConfig` [8]. verl ships native LoRA support for its RL trainers - `RayPPOTrainer` with `actor_rollout_ref.model.lora_rank`/`lora_alpha` set - on the FSDP/FSDP2 backend with vLLM or SGLang rollout, and separately on the Megatron backend [7].

## How it works

Each forward pass on a targeted layer runs the frozen base matrix and the low-rank correction in parallel and sums their outputs; only the correction gets gradients.

**The reparametrization** [1], Eq. 3: for a pre-trained weight matrix $W_0 \in \mathbb{R}^{d\times k}$, replace the update with a product of two low-rank matrices $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$, $r \ll \min(d,k)$:

$$ h = W_0 x + \Delta W x = W_0 x + BA\,x $$

$W_0$ is frozen throughout training; $A$ and $B$ hold the only trainable parameters, and $W_0 x$ and $BAx$ are computed separately and summed coordinate-wise [1]. $A$ is initialized with a random Gaussian and $B$ with zeros, so $\Delta W = BA$ is exactly zero at the start of training [1]. The update is scaled by $\alpha/r$ where $\alpha$ is a constant in $r$; the paper reports that tuning $\alpha$ with Adam is roughly equivalent to tuning the learning rate, so it sets $\alpha$ to the first value of $r$ it tries and does not retune it [1]. PEFT's implementation matches this scaling directly in code: the applied scale factor is `lora_alpha / r` by default, or `lora_alpha / sqrt(r)` when rank-stabilized LoRA is enabled [9] (read at PEFT `main`, 2026-08-07).

**Where it is applied**: the paper limits its study to the Transformer's four self-attention projections $W_q, W_k, W_v, W_o$, freezing the MLP module entirely for simplicity and parameter efficiency, and separately reports (Section 7.1) how the choice of which attention matrices to adapt changes results [1]. PEFT's `target_modules` accepts an explicit module list or the wildcard `all-linear`, which is not the paper's own choice [9]; verl's LoRA guide likewise recommends `target_modules` typically set to `all-linear` [7].

**Worked example**: for a GPT-3-scale matrix with $d=k=12{,}288$, the paper finds that a rank as low as $r=1$ or $r=2$ already suffices for strong downstream performance, even though the full matrix has rank up to 12,288 [1]. A rank-8 adapter on one such $12{,}288\times12{,}288$ matrix trains $B\,(12{,}288\times8)$ and $A\,(8\times12{,}288)$, about $2\times 12{,}288\times 8 \approx 196{,}608$ parameters, versus $12{,}288^2 \approx 151$M for the full matrix - roughly a 770x reduction on that one matrix. Consistent with that direction, the paper's own end-to-end figure for GPT-3 175B at $r=4$, adapting only $W_q$ and $W_v$, is a roughly 10,000x reduction in stored checkpoint size, from 350GB to 35MB [1].

**Merging for deployment**: because the correction is linear, $W = W_0 + BA$ can be computed once and stored, and the paper states that this construction means no inference latency is added relative to a model that was fully fine-tuned [1]. The paper notes this comes with a limitation: once $A$ and $B$ are merged for zero-latency inference, batching requests for different tasks (each with its own $A$, $B$) in a single forward pass is not straightforward, though it remains possible to keep the adapters unmerged and dynamically route batches when latency is not critical [1].

## Cost

**Theory, from the method's own math:**

- Time: LoRA adds one extra low-rank matrix multiply ($BAx$) per targeted layer on top of the frozen layer's own forward pass; because $r \ll d,k$ this multiply is small relative to $W_0 x$. Gradients and the backward pass are computed only for $A$ and $B$, not for the frozen majority of parameters, and the paper attributes a 25% training speedup on GPT-3 175B versus full fine-tuning directly to skipping that gradient computation [1] - an empirical, not a purely theoretical, number, kept here because the paper ties it directly to the reduced gradient computation.
- Memory: for a large Transformer trained with Adam, VRAM usage drops by up to 2/3 when $r \ll d_{model}$, because optimizer state is no longer needed for the frozen parameters [1]. On GPT-3 175B the paper reports this takes training VRAM from 1.2TB down to 350GB, and at $r=4$ with only $W_q,W_v$ adapted, checkpoint size drops from 350GB to 35MB (~10,000x) [1]. A naive reading might assume LoRA halves memory because "only two matrices are trained" - the actual saving is specifically the optimizer state (e.g., Adam's two per-parameter moment buffers [10]) for the frozen majority of weights, not the weights themselves, since $W_0$ is still held in memory for the forward pass [1].

**In practice, per framework:**

- PEFT [3][9]: `model.print_trainable_parameters()` reports the trainable fraction directly - the docs' own worked example for a BERT-scale model shows 667,493 trainable out of 86,543,818 total, 0.77% [3]. Weights `A` are Kaiming-uniform initialized and `B` zero-initialized by default, matching the reference implementation's zero-init convention [3].
- trl [8]: LoRA is attached via `peft_config`/`--use_peft` on top of an existing trainer (e.g., `SFTTrainer`); trl's own PEFT guide recommends raising the learning rate roughly 10x over the full-fine-tune rate when using LoRA (its worked example moves from `2.0e-5` to `2.0e-4`) [8] - a knob change, not a separate memory mechanism, since the underlying compute path is PEFT's.
- verl [7]: LoRA runs inside the RL actor on the FSDP/FSDP2 backend with vLLM or SGLang rollout, or on the Megatron backend; `rollout.load_format="safetensors"` is required so the rollout engine can load the base model, and `layered_summon=True` is recommended for 70B+ models or GPUs under 48GB to reduce peak memory when synchronizing the adapter to the rollout engine [7]. verl additionally offers `model.lora.merge`, which controls whether merged full weights or unmerged adapter deltas are transferred to the rollout engine - SGLang currently requires `merge=True` [7].

## How to use it

- Data prep is entirely the host method's: LoRA takes whatever prompts, labels, or preference pairs the wrapped trainer (SFT, DPO, PPO, GRPO, ...) already requires [1][7]. There is no LoRA-specific label format.
- Reward/label conventions: none of LoRA's own; inherited unchanged from the wrapped trainer.
- Key knobs, with each source's own default (paper values are the anchor, not a default anyone ships):

| knob | paper [1] | PEFT `LoraConfig` default [9] | trl `ModelConfig` (`--use_peft`) default [11] | verl (reference config) [7] |
| --- | --- | --- | --- | --- |
| rank $r$ | varies by experiment (as low as 1-2 on GPT-3) | `r` = 8 | `lora_r` = 16 | `lora_rank`, no library default - reference example uses 32 |
| $\alpha$ | set equal to first $r$ tried, not retuned | `lora_alpha` = 8 | `lora_alpha` = 32 | `lora_alpha`, no library default - reference example uses 32 |
| dropout | not stated | `lora_dropout` = 0.0 | `lora_dropout` = 0.05 | not checked |
| target modules | $W_q, W_k, W_v, W_o$ only, MLP frozen | `target_modules` = `None` (auto-detected by architecture) | `lora_target_modules` = `None` | `target_modules`, recommended `all-linear` |
| bias training | not stated | `bias` = `"none"` | not checked | not checked |

  Two readings of that table: PEFT's bare defaults (`r=8`, `alpha=8`, so scale $\alpha/r=1$) are conservative next to trl's (`r=16`, `alpha=32`, scale $2$) and verl's worked reference (`32`/`32`, scale $1$) - the ratio $\alpha/r$, not either number alone, sets the effective update magnitude [1][9][11]. And every framework here targets more of the network than the paper's attention-only default: PEFT auto-detects by architecture, trl and verl both point at `all-linear`-style targeting, while the paper explicitly froze the MLP module for its main results [1][7][9].
- Rank trade-off: verl's own guidance is that a very small `lora_rank` can lead to slower convergence or worse training performance, and its maintainers report near-parity with full RL training at `lora_rank>=32` for a 0.5B model and `lora_rank=128` for a 32B model, calling for more comprehensive reference results in the future [7]. Larger rank narrows the gap to full fine-tuning at the cost of more trainable parameters and memory.
- Learning rate: trl's guide pairs LoRA with a roughly 10x higher learning rate than the corresponding full-fine-tune run, and verl's guide separately recommends increasing the learning rate "by an order of magnitude" [8][7] - two independent frameworks converging on the same direction of adjustment.

## While it runs

- Signals and their healthy shapes: neither the paper nor the framework sources checked here name a LoRA-specific training curve to watch beyond the wrapped method's own signals (loss, task metric, or - inside an RL loop - reward and entropy as covered by that method's card); LoRA changes what is trained, not what is logged.
- Published reference runs: verl's LoRA guide includes reward/throughput curves from a Qwen3-8B run on 8xH200 comparing the FSDP and Megatron backends, alongside a separate rank-sweep chart from its GitHub community assets [7].
- Degeneracies and defaults: setting `target_modules` too narrowly (attention-only, as the paper's own default) while running a modern all-linear-tuned recipe silently gives less capacity than the framework's typical usage - check which modules are actually targeted rather than assuming a default [1][9]. `lora_alpha` and `r` interact only through their ratio; changing one without the other silently rescales every update [1][9].
- Named successors: LoRA+ separates the learning rates of the $A$ and $B$ matrices [5]; rank-stabilized LoRA (rsLoRA) changes the scaling factor to $\alpha/\sqrt{r}$ instead of $\alpha/r$, an option PEFT ships directly as `use_rslora` [9].
- Known failure modes: the paper's own limitations section states that batching inputs for different tasks with different merged $A,B$ pairs in a single forward pass is not straightforward once weights are merged for zero-latency inference [1]. A later empirical study (not the defining paper) reports that, in standard low-rank settings, LoRA substantially underperforms full fine-tuning on programming and mathematics instruction-tuning and continued-pretraining regimes, and traces part of the gap to full fine-tuning learning weight perturbations with an effective rank 10-100x larger than typical LoRA configurations [12]. No maintainer-issue-level failure mode search was performed for this card.
- What the gain is - and is not: the same study finds that despite the gap on the target domain, LoRA better preserves the base model's performance outside that domain and mitigates forgetting more than weight decay or dropout, while also maintaining more diverse generations than full fine-tuning [12]. The gain is compute/memory efficiency and reduced forgetting on off-target tasks; it is not, at typical ranks, full fine-tuning's peak in-domain performance [12].

## Sources

[1] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. https://arxiv.org/abs/2106.09685 - defines LoRA: reparametrization, scaling, application to Transformer attention weights, memory/latency analysis, GPT-3 175B results (Table 4), rank-deficiency investigation, limitations. Fetched 2026-08-07 (PDF full text via arxiv.org/pdf/2106.09685).

[2] Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", 2019. https://arxiv.org/abs/1902.00751 - adapter tuning, the nearest earlier alternative LoRA contrasts itself against. Fetched 2026-08-07 (abstract).

[3] Hugging Face PEFT, LoRA documentation. https://huggingface.co/docs/peft/main/en/package_reference/lora - `LoraConfig`/`get_peft_model` quick-start, trainable-parameter example, default initialization scheme. Fetched 2026-08-07.

[4] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023. https://arxiv.org/abs/2305.14314 - landmark adopter, 4-bit quantized backprop into LoRA adapters. Fetched 2026-08-07 (abstract).

[5] Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models", 2024. https://arxiv.org/abs/2402.12354 - named successor: shows LoRA's shared learning rate for adapter matrices A and B is suboptimal for large-width models and proposes separate rates. Fetched 2026-08-07 (abstract).

[6] Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA", 2023. https://arxiv.org/abs/2312.03732 - rank-stabilized LoRA scaling ($\alpha/\sqrt{r}$), shipped in PEFT as `use_rslora` per [9]. Fetched 2026-08-07 (abstract).

[7] verl documentation, "RL(HF) algorithms with LoRA Support". https://raw.githubusercontent.com/volcengine/verl/main/docs/advance/ppo_lora.rst - native LoRA support for PPO/GRPO actors, FSDP/FSDP2 and Megatron backends, config keys, rank guidance, reference run images. Fetched 2026-08-07 (raw doc source, page dated "Last updated: 02/03/2026").

[8] trl documentation, "PEFT Integration". https://huggingface.co/docs/trl/main/en/peft_integration - `peft_config`/`--use_peft`, recommended 10x learning-rate increase for LoRA. Fetched 2026-08-07.

[9] PEFT source, `src/peft/tuners/lora/config.py` and `src/peft/tuners/lora/layer.py`, `main` branch. https://github.com/huggingface/peft - `LoraConfig` field defaults (`r=8`, `lora_alpha=8`, `lora_dropout=0.0`, `bias="none"`, `target_modules=None`), scaling factor computation (`lora_alpha/r` or `lora_alpha/sqrt(r)` under `use_rslora`). Fetched 2026-08-07 at `main` (commit not pinned by this fetch; code read live from the default branch).

[10] Kingma and Ba, "Adam: A Method for Stochastic Optimization", 2015. https://arxiv.org/abs/1412.6980 - the two per-parameter moment estimates behind the optimizer-state memory claim. Fetched 2026-08-07 (abstract).

[11] trl source, `trl/trainer/model_config.py`, `main` branch. https://github.com/huggingface/trl - `ModelConfig` LoRA field defaults (`lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`). Fetched 2026-08-07 at `main` (commit not pinned by this fetch).

[12] Biderman et al., "LoRA Learns Less and Forgets Less", 2024. https://arxiv.org/abs/2405.09673 - LoRA-vs-full-fine-tuning performance gap on programming/math, effective-rank explanation, and reduced forgetting/more diverse generations outside the target domain. Fetched 2026-08-07 (abstract).
