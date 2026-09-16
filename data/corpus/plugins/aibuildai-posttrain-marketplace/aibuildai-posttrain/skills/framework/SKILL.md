---
description: >-
  Libraries that post-train a base language model, and how to run one. 108 library
  cards, each covering when to pick that library, how to start the run, what to watch
  while it runs, and how to save a result something else can load. The body of this
  skill holds a short recommended table; references/index.md holds every library with
  the methods measured in its own code, its live-or-archived state, and the date it was
  last pushed. references/loading-the-result.md is the loader-side contract and is read
  before the first save. Use it when choosing where to run a post-training job, and
  again while that job runs.
---

# Post-training libraries

A post-training library is the code that runs the training: it holds the trainer, the data loader, the distributed launch, and the checkpoint writer, so the method you pick is only a setting inside it. This skill serves two moments, and one card serves both: choosing which library to run this training in, and then operating that run - start it, watch it, save it. There is no separate training skill. The choice is usually decided by which methods a library ships and whether it is still alive, not by which one is famous.

## When to consult this skill

- The designer, when the plan names a training method and the run needs somewhere to run it.
- The implementer, before writing the first training script: the card carries the install line, the launch command, and the config class.
- The implementer again while the run is live: the same card says what to watch and what each signal means.
- Anyone about to save a checkpoint an evaluator must load later: read `references/loading-the-result.md` first.

## Recommended libraries

A short table, not the full list: 16 libraries that cover the common cases. Every other library is in `references/index.md`, which carries all 108 rows with the same measured facts.

| Library | When to pick it | Methods seen in its code | State | Models on the Hub | File |
|---|---|---|---|---|---|
| `trl` | post-training in the Hugging Face ecosystem, when your models and datasets live on the Hub and you want a trainer-class API that starts on one machine - a wide method menu (16 trainers) ... | DPO, GRPO, KTO, ORPO, PPO, REWARD, SFT | live, pushed 2026-07-31 | 156,252 | `huggingface__trl.md` |
| `verl` | RL post-training that needs to scale across multiple nodes with an explicit cluster scheduler ... | DAPO, DPO, PPO, REWARD, SFT | live, pushed 2026-07-31 | 191 | `verl-project__verl.md` |
| `LlamaFactory` | pick LlamaFactory when the priority is fine-tuning a very wide range of already-published LLM/VLM checkpoints (the paper's title claims support for over 100 language models) through a fixed ... | DPO, KTO, PPO, SFT, Trainer | live, pushed 2026-07-31 | 0 | `hiyouga__LlamaFactory.md` |
| `ms-swift` | fine-tuning or RLHF on models pulled from the ModelScope hub (or Hugging Face, via `--use_hf true`), when you want one CLI to cover LoRA/full-parameter SFT, RLHF (DPO/KTO/GRPO/PPO/RM and more) ... | GRPO, KTO, REWARD, RLHF | live, pushed 2026-07-31 | 56 | `modelscope__ms-swift.md` |
| `unsloth` | single-consumer-GPU fine-tuning or RL where VRAM is the binding constraint ... | KTO, ORPO | live, pushed 2026-07-31 | 139,196 | `unslothai__unsloth.md` |
| `peft` | peft is the adapter layer under most Hugging Face post-training stacks ... | SFT | live, pushed 2026-07-31 | 259,817 | `huggingface__peft.md` |
| `transformers` | as a post-training tool, pick `transformers.Trainer` directly when the job is a plain supervised loss on a Hub model and dataset and you don't need a preference, RLHF, or distillation trainer ... | Trainer | live, pushed 2026-07-31 | not measured | `huggingface__transformers.md` |
| `axolotl` | you want one YAML file to drive the whole pipeline - model choice, method choice (SFT, DPO/IPO/KTO/SimPO/ORPO, GRPO, EBFT, or reward/PRM training), sharding strategy, and dataset format ... | DPO, GRPO, KTO, ORPO, PPO, REWARD, SFT, Trainer | live, pushed 2026-07-31 | 49,865 | `axolotl-ai-cloud__axolotl.md` |
| `torchtune` | pick torchtune only for a PyTorch-native, single-file-recipe way to fine-tune Llama/Qwen/Gemma-family checkpoints on a fixed, working method set ... | DPO, GRPO, PPO, RLHF | live, pushed 2026-07-31 | 38 | `meta-pytorch__torchtune.md` |
| `OpenRLHF` | online RLVR/agentic RL at cluster scale, when the workload needs Ray to place and time-slice Actor/Critic/Reward/Reference/vLLM across GPUs and you want algorithm choice ... | DPO, GRPO, KTO, PPO, REWARD, SFT | live, pushed 2026-07-14 | not measured | `OpenRLHF__OpenRLHF.md` |
| `open-instruct` | reproducing or extending AI2's own Tulu/OLMo post-training recipes (SFT, DPO, reward modeling, GRPO with verifiable rewards) ... | DPO, GRPO, REWARD, SFT | live, pushed 2026-07-31 | 11 | `allenai__open-instruct.md` |
| `alignment-handbook` | you want the exact, published, reproducible recipe (YAML config plus launch command) behind a specific released model — Zephyr-7B-β, StarChat2 ... | DPO | live, pushed 2026-05-26 | 4,998 | `huggingface__alignment-handbook.md` |
| `ColossalAI` | pick it when the parallelism strategy itself is the reason to choose a framework ... | GRPO, KTO, ORPO | live, pushed 2026-07-13 | 0 | `hpcaitech__ColossalAI.md` |
| `AReaL` | pick AReaL when the run needs the generation and training steps decoupled and independently scaled through named services ... | DPO, PPO, SFT | live, pushed 2026-07-31 | 3 | `areal-project__AReaL.md` |
| `slime` | RL post-training at Megatron-training / SGLang-rollout scale ... | PPO, SFT | live, pushed 2026-07-24 | 6 | `THUDM__slime.md` |
| `LMFlow` | a single, script-driven toolbox for LLM finetuning and alignment that bundles memory-saving finetuning (LISA, LoRA, QLoRA) with a curated set of alignment methods ... | DPO, REWARD | live, pushed 2026-05-22 | 0 | `OptimalScale__LMFlow.md` |

Rules for reading this table, and each has a reason behind it:

- **The when-to-pick cell is one sentence, compressed from the card's own verdict.** Everything beyond it is on the card, never in a second index.
- **`Methods seen in its code` is measured in the repository**, by finding the trainer class or the algorithm setting, not by reading a claim in a README. A library may support more than its code search showed; the card says what it really ships.
- **This table is filtered, not ranked.** Pick by which methods a library ships and whether it is alive. Two popularity numbers were tried as a ranking and both failed, which is why neither is a column here. A star count is not comparable across repositories, because a mirror can carry a different count from the project's own home. And the Hub model count is not use either: `LlamaFactory` reads 0 against 73,649 stars, `DeepSpeed` reads 0 against 42,842, and `ColossalAI` reads 0 against 41,426. The count records whether a library's name became a Hub tag, which is a habit of the transformers toolchain.
- **The Hub count comes from the exact tag filter** `/api/models?filter=<library>`, never from the loose `?search=` form, which matches any model id containing those letters. Many library names are ordinary English words, so the loose form is wrong by orders of magnitude, not by a little.
- **State is read at a commit and dated**, because discovery never reveals an archived repository. A list without this column would recommend a dead library.

## How to use a card

Read one card, not all of them. They share an outline, so two can be read side by side to compare, and each carries both the choice and the operation of its library: quick start, start it, watch it, save it, and where to find its docs. The one cross-library file is `references/loading-the-result.md` - the loader side's contract on a saved model, which holds whatever library trained it. Read it before the first save, not after the run.

## The full list

`references/index.md` holds all 108 libraries with every measured fact, and `references/index.json` holds the same rows for matching by field.

## Hosted services

`references/hosted-services.md` covers post-training that runs as a service, with no repository to clone. It is thin on purpose: only a provider with a first-hand source is listed, and the file names the providers that were checked and left out.

## Scope

- Libraries published on github.com. A project found only on Gitee, GitCode, or ModelScope is out of scope by choice, not by accident.
- The line is the platform, not the authors: `ms-swift` is here because its repository is on GitHub.
- **This list is not saturated, and that is measured.** GitHub code search reports far more results than it will hand over: tens of thousands of counted results were never returned. Every one of the nine channels still found libraries no other channel found, and the curated-list channel alone brought in 500 of them. A saturated list would show neither.
- A library with no card is not a library that does not exist. `references/index.md` says what was measured and what was not.
- Every claim on a card carries a URL and a commit or a fetch date. Nothing comes from our own runs.
