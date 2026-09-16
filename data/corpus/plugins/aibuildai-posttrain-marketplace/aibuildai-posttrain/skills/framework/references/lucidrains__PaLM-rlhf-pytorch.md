# PaLM-rlhf-pytorch

A single-author research implementation of RLHF on top of a from-scratch PaLM transformer: plain PyTorch classes, no Trainer framework, no distributed launcher templates - read the source before you run it.

**PaLM-rlhf-pytorch** is described by its own README as "Implementation of RLHF (Reinforcement Learning with Human Feedback) on top of the PaLM architecture" [1]. It is written and maintained by Phil Wang (lucidrains) [2], and it lives at https://github.com/lucidrains/PaLM-rlhf-pytorch [3]. The API is three hand-built `torch.nn.Module` classes wired together by hand in user code: a `PaLM` transformer, a `RewardModel` built on a cloned `PaLM`, and a method-specific trainer (e.g. `RLHFTrainer` for PPO) that takes the two models plus a tensor or file of prompt token ids and runs `.train(num_episodes=...)` [1].

**When to pick it**: a from-scratch, readable reference for how PPO/GRPO/TPO/FlowRL-style RLHF loops are wired around a PaLM-style transformer, or as a small-scale sandbox to read and modify the RL loop directly - not a production or cluster-scale trainer. It has no Hub integration, no dataset-format abstraction, no tracker logging out of the box, and no documented multi-node story, unlike `trl` or `verl` (cross-reference; not covered here), which are the picks once you need to train a real pretrained checkpoint at scale or want built-in experiment tracking.

**Methods it ships**: `RLHFTrainer` (PPO) plus `ActorCritic` in `palm_rlhf_pytorch/ppo.py`, exported at the top level as `from palm_rlhf_pytorch import RLHFTrainer, ActorCritic` [4]. GRPO, TPO (Target Policy Optimization), and FlowRL each live in their own module - `palm_rlhf_pytorch/grpo.py`, `palm_rlhf_pytorch/tpo.py`, `palm_rlhf_pytorch/flowrl.py` - and each defines its own class also named `RLHFTrainer` (FlowRL's is `FlowRLTrainer`); none of these three is re-exported from the package `__init__.py`, so they must be imported from their submodule path, e.g. `from palm_rlhf_pytorch.grpo import RLHFTrainer` [4][5][6][7]. GRPO's constructor additionally gates three cited variants behind flags on the same class: `use_simple_policy_optimization`, `use_dr_grpo`, and `use_max_rl` [5]. A separate `RewardModel` (`palm_rlhf_pytorch/reward.py`) and an `ImplicitPRM` process-reward module (`palm_rlhf_pytorch/implicit_process_reward.py`, exported at top level) round out the method surface [8][9][4]. There is no methods index page; this list is read directly from the module tree at the reviewed commit and can drift - recheck the repository file listing before relying on it [10].

**Scale it handles**: single GPU is the only pattern shown in the README and in `examples.py` - both call `.cuda()` or `Accelerator().device` directly on one process [1][11]. Each trainer builds its own `accelerate.Accelerator(**accelerate_kwargs)` internally [4][5], so multi-GPU data parallelism is mechanically reachable by launching the script with `accelerate launch`, but no README section, example script, or accelerate config template in the repository demonstrates or benchmarks multi-GPU or multi-node training, and there is no DeepSpeed or FSDP sharding integration anywhere in the source [1][10]. Treat any claim beyond single-GPU as unverified mechanism, not a documented or measured capability.

**Install**: `pip install palm-rlhf-pytorch`; version 0.7.5, uploaded to PyPI 2026-07-27T23:39:49Z [12]; `requires-python >= 3.6` [13]; MIT licence [1][13]. Dependencies carry only lower-bound floors, no upper caps: `torch>=2.2` is the only pinned deep-learning core, alongside `accelerate`, `adam-atan2-pytorch`, `beartype`, `einx>=0.3.0`, `einops>=0.8.0`, `memmap-replay-buffer>=0.1.4`, `torch-einops-utils`, `hl-gauss-pytorch>=0.1.19`, `tqdm`, and `x-mlps-pytorch`, all unpinned above their floor - see `pyproject.toml` at the reviewed commit for the exact list [13]. No git tag or GitHub Release exists for 0.7.2-0.7.5 (the newest tagged release is `0.7.1`, published 2025-09-19) [14][15]; this card instead reads the repository's newest push, commit `bd0d0f3`, pushed 2026-07-27T23:39:41Z - eight seconds before the matching PyPI upload timestamp - and whose `pyproject.toml` already reads `version = "0.7.5"` [16][13], so the commit and the PyPI release are treated as the same build. No CUDA or GPU minimum is stated anywhere in the docs or package metadata; a maintainer reply to "Do you need cuda for this?" confirms CUDA is required in practice ("yes to title") [17].

**Maintained by**: Phil Wang (lucidrains), a single maintainer; about 7,864 GitHub stars (not a ranking signal) [3]. The repository is active - last push 2026-07-27T23:39:41Z and a PyPI upload the same second [16][12] - but the README itself is still marked as work-in-progress, and its FAQ says no trained model is available and that producing one would take an outlay on the order of millions of dollars in compute and data [1]. The README also names Direct Preference Optimization as a likely successor that would collapse most of this repository's PPO/reward-model machinery into a much smaller loss, i.e. the maintainer's own view that this design may be superseded [1].

## Quick start

The three README snippets, verbatim in shape, chain into one working (mock-data) script; the same sequence appears as a real runnable file at `examples.py` [11][1]:

```python
import torch
from palm_rlhf_pytorch import PaLM

palm = PaLM(
    num_tokens = 20000,
    dim = 512,
    depth = 12,
    flash_attn = True
).cuda()

seq = torch.randint(0, 20000, (1, 2048)).cuda()
loss = palm(seq, return_loss = True)
loss.backward()

generated = palm.generate(2048)  # (1, 2048)
```

```python
import torch
from palm_rlhf_pytorch import PaLM, RewardModel

palm = PaLM(num_tokens = 20000, dim = 512, depth = 12, causal = False)

reward_model = RewardModel(
    palm,
    num_binned_output = 5  # say rating from 1 to 5
).cuda()

seq = torch.randint(0, 20000, (1, 1024)).cuda()
prompt_mask = torch.zeros(1, 1024).bool().cuda()
labels = torch.randint(0, 5, (1,)).cuda()

loss = reward_model(seq, prompt_mask = prompt_mask, labels = labels)
loss.backward()

reward = reward_model(seq, prompt_mask = prompt_mask)
```

```python
import torch
from palm_rlhf_pytorch import PaLM, RewardModel, RLHFTrainer

palm = PaLM(num_tokens = 20000, dim = 512, depth = 12).cuda()
palm.load('./path/to/pretrained/palm.pt')

reward_model = RewardModel(palm, num_binned_output = 5).cuda()
reward_model.load('./path/to/pretrained/reward_model.pt')

prompts = torch.randint(0, 256, (50000, 512)).cuda()  # 50k prompts

trainer = RLHFTrainer(
    palm = palm,
    reward_model = reward_model,
    prompt_token_ids = prompts
)

trainer.train(num_episodes = 50000)

answer = trainer.generate(2048, prompt = prompts[0], num_samples = 10)
```

There is no CLI entry point anywhere in the repository - every run is a Python script [10].

## Start it

- One process, one GPU is the only pattern the README, `examples.py`, and `train.py` (base PaLM pretraining) demonstrate; `train.py` wraps the loop in `accelerate.Accelerator(gradient_accumulation_steps=...)` for a single accelerate-managed process [1][11][18].
- More than one GPU has no documented config template or example in this repository; the internal `Accelerator(**accelerate_kwargs)` each trainer builds is the only mechanism, reachable in principle by running the script under `accelerate launch` after `accelerate config`, but that path is unverified here since neither the README nor any example exercises it [4][5][10].
- Batch arithmetic is set by trainer constructor and `train()` arguments rather than a single global batch size: PPO's `RLHFTrainer` takes `minibatch_size` (default 16) and `epochs` (default 1) at construction, and `num_episodes`, `max_timesteps`, `update_timesteps`, `max_batch_size` at `.train()` time [19]. GRPO's constructor additionally exposes `grpo_num_times_sample_rewards` (default 10, the group size) [5].
- Configuration is plain constructor keyword arguments on each trainer class - there is no separate Config object. None of the defaults set a training precision (no bf16/fp16 flag appears in any trainer constructor), so precision follows whatever the surrounding `Accelerator` or manual `.cuda()`/`.half()` calls do - this library sets no silent precision default of its own [4][5][19].
- Out-of-memory first aid is not published anywhere in the README or source as a dedicated section; the closest lever is lowering `minibatch_size` and `max_batch_size`/`max_timesteps` in the constructor and `.train()` call shown above, since there is no separate generation engine or documented memory knob to tune [19][10].

## Watch it

This section covers only the mechanics of what this repository emits; no method-level interpretation is given here.

- There is no tracker integration anywhere in the source: no `wandb`, no `tensorboard`, and no `report_to`-style field exist in `ppo.py`, `grpo.py`, `tpo.py`, or `flowrl.py` - confirmed by searching all four files for `wandb`, logging calls, and checkpoint helpers [19][5][6][7]. The only run-time signal is `self.print(...)`, a thin wrapper around `Accelerator.print` that writes to the console (and to any tracker the surrounding script's own `Accelerator` happens to have configured, which none of the shipped scripts do) [19].
- PPO's `RLHFTrainer` prints `policy_loss` and `critic_loss` each optimization step [19]. GRPO's `RLHFTrainer` prints `policy_loss` [5]. TPO's `RLHFTrainer` prints `policy_loss` [6]. `FlowRLTrainer` prints `flowrl_loss` and `trajectory_balance` [7]. Each ends its `.train()` with a one-line completion message ("rlhf training complete", "dr grpo rlhf training complete", "tpo training complete", "FlowRL training complete") [19][5][6][7].
- There is no sample-level generation logging (no flag to periodically print or save generated completions during RLHF training) in any of the four trainer files [19][5][6][7]. `train.py`, the separate base-PaLM pretraining script, does print a generated sample every `GENERATE_EVERY` steps, but that script trains the base transformer, not RLHF [18].
- There is no built-in evaluation-during-training path (no `eval_dataset` or `eval_steps`-style field) in any trainer constructor [19][5][6][7].
- No stopping-rule, early-stopping, or divergence-threshold field is published in any of the four trainer files - confirmed by grepping all four for `stop`, `threshold`, and `patience`-style names, which return no matches [19][5][6][7].

## Save it

- `RLHFTrainer.save(filepath='./checkpoint.pt')` (PPO) writes `torch.save(self.actor_critic.state_dict(), filepath)` - the full `ActorCritic` module, which bundles the actor PaLM (with its LoRA finetune parameters as extra sub-modules of the same object, not a separate adapter file) and the critic together in one state dict [19]. `RLHFTrainer.load(filepath)` reloads that same state dict into `self.actor_critic` [19]. GRPO, TPO, and FlowRL each save/load only their own `self.actor` state dict in the same pattern [5][6][7].
- None of the four `save`/`load` pairs persists optimizer or scheduler state, so `load()` restores model weights only - there is no resume-from-checkpoint call that also restores optimizer momentum or step count; confirmed by reading all four `save`/`load` method bodies, which call `state_dict()`/`load_state_dict()` on the model alone [19][5][6][7].
- `PaLM.load(path)` and `RewardModel.load(path)` are the two base-model loaders (`self.load_state_dict(torch.load(str(path)))`); neither class defines its own `.save()` method, so saving a `PaLM` or `RewardModel` on its own is a plain `torch.save(model.state_dict(), path)` in user code, as implied by the README's load-only usage of pretrained weights [20][8][1].
- LoRA in this repository is not a separate adapter checkpoint format: `add_finetune_params(scope, lora_r=...)` registers the LoRA weights as named sub-modules inside the same `PaLM`/`ActorCritic` object, so every save above is a full-model state dict that already includes any LoRA parameters, not a small standalone adapter directory [20][19].
- Loader handoff: because every checkpoint here is a raw PyTorch `state_dict`, loading it back requires reconstructing the exact same class (`PaLM`, `RewardModel`, or `ActorCritic`) with matching constructor arguments and calling `.load_state_dict()` - there is no `from_pretrained`-style loader, no Hub push, and no format an external evaluator can load without first re-instantiating this repository's own classes [20][19][1].

## Find it in the docs

There is no separate documentation site for this repository - the GitHub README is the only prose documentation, and the module source is the only reference for constructor arguments and defaults [1][10].

- Item home: https://github.com/lucidrains/PaLM-rlhf-pytorch [3]. The `pyproject.toml` `Homepage` field instead points to a Codeberg mirror, `https://codeberg.org/lucidrains/PaLM-rlhf-pytorch` [13]; this card treats the GitHub repository as the primary source because that is where the commit and issue history used throughout this card live.
- For any constructor argument or default not covered by this card, open the relevant module directly: `palm_rlhf_pytorch/ppo.py` (PPO `RLHFTrainer`, `ActorCritic`), `palm_rlhf_pytorch/grpo.py` (GRPO `RLHFTrainer`), `palm_rlhf_pytorch/tpo.py` (TPO `RLHFTrainer`), `palm_rlhf_pytorch/flowrl.py` (`FlowRLTrainer`), `palm_rlhf_pytorch/reward.py` (`RewardModel`), `palm_rlhf_pytorch/implicit_process_reward.py` (`ImplicitPRM`), `palm_rlhf_pytorch/lora.py` (`LoRA`), `palm_rlhf_pytorch/palm.py` (`PaLM`) [4][19][5][6][7][8][9][21][20].
- Runnable references beyond the README: `examples.py` at the repository root runs the full PPO README sequence end to end on mock data and is the closest thing to a smoke test in the repository [11]; `train.py` pretrains the base `PaLM` on the bundled `data/enwik8.gz` corpus, a known-good small dataset for exercising the base transformer only, not the RLHF loop [18][10].
- Community layer: none - the README's Community section links to CarperAI's separate `trlx` project and Yannic Kilcher's separate Open-Assistant project as related efforts, not as tutorials on this repository, and lists three YouTube videos about RLHF in general rather than this codebase specifically [1]. There is no official MCP endpoint for this repository.
- Known trap, from a closed issue: issue #66, "Bug when training reward model in examples.py", was acknowledged by the maintainer (association: OWNER) on 2025-08-26 with a fix landing in version 0.5.5 - if reproducing the reward-model quickstart on an older pinned version, upgrade past 0.5.5 first [22].
- Honest boundary: this repository trains from a from-scratch `PaLM` implementation, not a pretrained Hub checkpoint - the README's FAQ says no trained model is available and that reaching a useful one would need an outlay on the order of millions of dollars in compute and data [1]; there is no dataset-format abstraction, no Hub integration, and (per Scale it handles, above) no documented multi-GPU or multi-node run.

## Sources

All GitHub source files are cited at commit `bd0d0f3893ec62f827fa35e759cb38d748829d51`, the repository's newest push as of the review date; see the Install field above for why this commit is treated as equivalent to the 0.7.5 PyPI release. All pages were fetched or read 2026-08-10.

[1] PaLM-rlhf-pytorch README. https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/README.md. Read at commit bd0d0f3, fetched 2026-08-10.

[2] PyPI project page for palm-rlhf-pytorch (author metadata). https://pypi.org/project/palm-rlhf-pytorch/. Fetched 2026-08-10.

[3] PaLM-rlhf-pytorch GitHub repository (item home). https://github.com/lucidrains/PaLM-rlhf-pytorch. Fetched 2026-08-10 via the GitHub API (`api.github.com/repos/lucidrains/PaLM-rlhf-pytorch`).

[4] `palm_rlhf_pytorch/__init__.py` (top-level export list). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/palm_rlhf_pytorch/__init__.py. Read at commit bd0d0f3, fetched 2026-08-10.

[5] `palm_rlhf_pytorch/grpo.py` (GRPO `RLHFTrainer`, its `use_simple_policy_optimization`/`use_dr_grpo`/`use_max_rl` flags, save/load, printed metrics). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/palm_rlhf_pytorch/grpo.py. Read at commit bd0d0f3, fetched 2026-08-10.

[6] `palm_rlhf_pytorch/tpo.py` (TPO `RLHFTrainer`, save/load, printed metrics). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/palm_rlhf_pytorch/tpo.py. Read at commit bd0d0f3, fetched 2026-08-10.

[7] `palm_rlhf_pytorch/flowrl.py` (`FlowRLTrainer`, save/load, printed metrics). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/palm_rlhf_pytorch/flowrl.py. Read at commit bd0d0f3, fetched 2026-08-10.

[8] `palm_rlhf_pytorch/reward.py` (`RewardModel`, its `load` method). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/palm_rlhf_pytorch/reward.py. Read at commit bd0d0f3, fetched 2026-08-10.

[9] `palm_rlhf_pytorch/implicit_process_reward.py` (`ImplicitPRM`). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/palm_rlhf_pytorch/implicit_process_reward.py. Read at commit bd0d0f3, fetched 2026-08-10.

[10] Repository file tree at commit bd0d0f3. https://api.github.com/repos/lucidrains/PaLM-rlhf-pytorch/git/trees/bd0d0f3893ec62f827fa35e759cb38d748829d51?recursive=1. Fetched 2026-08-10.

[11] `examples.py` (root-level runnable PPO quickstart script). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/examples.py. Read at commit bd0d0f3, fetched 2026-08-10.

[12] PyPI JSON API for palm-rlhf-pytorch (version and upload timestamp). https://pypi.org/pypi/palm-rlhf-pytorch/json. Fetched 2026-08-10.

[13] `pyproject.toml` at commit bd0d0f3 (version, licence, Python floor, dependency floors, Homepage URL). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/pyproject.toml. Read at commit bd0d0f3, fetched 2026-08-10.

[14] GitHub tags list. https://api.github.com/repos/lucidrains/PaLM-rlhf-pytorch/tags. Fetched 2026-08-10.

[15] GitHub releases list. https://api.github.com/repos/lucidrains/PaLM-rlhf-pytorch/releases. Fetched 2026-08-10.

[16] GitHub repository metadata (`pushed_at` timestamp for commit bd0d0f3). https://api.github.com/repos/lucidrains/PaLM-rlhf-pytorch. Fetched 2026-08-10.

[17] Closed issue #30, "Do you need cuda for this?" - maintainer (lucidrains, OWNER) reply "yes to title, no to the body", 2023-02-24. https://api.github.com/repos/lucidrains/PaLM-rlhf-pytorch/issues/30/comments. Fetched 2026-08-10.

[18] `train.py` (base-PaLM pretraining script, single-process `Accelerator` usage, periodic sample generation). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/train.py. Read at commit bd0d0f3, fetched 2026-08-10.

[19] `palm_rlhf_pytorch/ppo.py` (PPO `RLHFTrainer`, `ActorCritic`, constructor defaults, save/load, printed metrics, completion message). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/palm_rlhf_pytorch/ppo.py. Read at commit bd0d0f3, fetched 2026-08-10.

[20] `palm_rlhf_pytorch/palm.py` (`PaLM` class, its `load` method, `add_finetune_params` for LoRA scopes). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/palm_rlhf_pytorch/palm.py. Read at commit bd0d0f3, fetched 2026-08-10.

[21] `palm_rlhf_pytorch/lora.py` (`LoRA` module). https://raw.githubusercontent.com/lucidrains/PaLM-rlhf-pytorch/bd0d0f3893ec62f827fa35e759cb38d748829d51/palm_rlhf_pytorch/lora.py. Read at commit bd0d0f3, fetched 2026-08-10.

[22] Closed issue #66, "Bug when training reward model in examples.py" - maintainer (lucidrains, OWNER) reply asking to retry on 0.5.5, 2025-08-26. https://api.github.com/repos/lucidrains/PaLM-rlhf-pytorch/issues/66/comments. Fetched 2026-08-10.
