# DI-engine

A general-purpose deep-RL engine built around a python-first, asynchronous-native "task/middleware" pipeline, with GRPO and LLM-preference-data utilities recently added on its unreleased main branch, not in its shipped PyPI release.

DI-engine states of itself: "DI-engine is a generalized decision intelligence engine for PyTorch and JAX," providing "python-first and asynchronous-native task and middleware abstractions" that modularly integrate three core concepts - Env, Policy and Model [1]. It is built and maintained by OpenDILab, credited in its own citation entry as Niu, Xu, Pu, Nie, Zhang, Hu, Zhao, Zhang and Liu, with a contact address at pjlab.org.cn [1]. The pipeline shape is: define an env manager, a `Policy` (wrapping a `Model`), assemble `task.use(...)` middleware functions (collector, evaluator, trainer, checkpoint saver, logger) inside a `with task.start(ctx=...)` block, then call `task.run()` [2]. It lives at https://github.com/opendilab/DI-engine [3].

**When to pick it**: a classic/general deep-RL engine first - the README's own algorithm list runs to single/multi-agent DRL, imitation learning, offline RL, model-based RL and exploration algorithms, with "LLM + RL Algorithms: PPO-max, DPO, PromptPG, PromptAWR" as one category among many, not the library's focus [1]. Its LLM-facing pieces are narrow and, per the pinned-vs-live check below, not yet in the released package: a GRPO loss function (`ding/rl_utils/grpo.py`) and an `OfflineRLDataset` for KTO/DPO-style preference data (`ding/utils/data/rlhf_offline_dataset.py`), both added to the repository after the last PyPI release [4][5]. There is no `ding/policy/dpo.py` or comparable DPO trainer at the screening commit - a 404 on that path confirms it - so "DPO" in the README's algorithm list is a name in prose, not a shipped, linked implementation the way PPO and PromptPG are (both carry a doc link and a `ding/policy/*.py` source path in the README's algorithm table) [1][6]. For post-training work built around trainer classes and Hub models/datasets, weigh the trl or verl cards instead (cross-reference; not covered here).

**Methods it ships**: DI-engine groups its methods in the README's "Algorithm Versatility" table, which lists a doc link and source path per algorithm (57 rows read) - among them PPO/MAPPO at `ding/policy/ppo.py` and PromptPG at `ding/policy/prompt_pg.py`, both with doc pages [1]. GRPO is not in that table; it lives only as a standalone loss function, `grpo_policy_error(data, log_prob_fn=..., clip_ratio=0.2, beta=0.1)` in `ding/rl_utils/grpo.py`, citing arXiv 2402.03300, and returning a loss plus `(approx_kl, clipfrac)` diagnostics - it is a per-token-logprob loss utility a caller wires into their own training loop, not a `Trainer` class [4]. RLHF/preference-data support is `OfflineRLDataset` in `ding/utils/data/rlhf_offline_dataset.py`, whose docstring states it is a "PyTorch Dataset for OfflineRL LLM training like KTO and DPO," turning `(input, output, label)` records into tokenized `(prompt, response, label, prompt_ids_len)` items via a `transformers.AutoTokenizer` [5] - it produces training data, not a training loop. The generic `trainer`/`multistep_trainer` middleware functions in `ding/framework/middleware/functional/trainer.py` call `policy.forward(ctx.train_data)` and log a scalar loss; the file ends with a bare `# TODO reward model` comment, meaning no reward-model middleware ships there yet [7]. The taxonomy moves - re-check the README's algorithm table at the live page for additions [1].

**Scale it handles**: single-process, single-GPU is the base case (`python3 -u ding/example/ppo.py`) [2]. Multi-process/multi-worker runs use the `ditask` CLI, e.g. `ditask --package . --main ding.example.ppo.main --parallel-workers 4 --topology mesh`, splitting one learner + one evaluator + N collectors across processes connected by a "mesh" topology, with roles assigned via `task.add_role(...)` and state synced through `ContextExchanger`/`ModelExchanger` middleware [2]. A dedicated docs page also documents DDP-based distributed training (`04_best_practice/ddp.html`) [8]. None of this is LLM-scale (no FSDP/DeepSpeed sharding, no vLLM generation layer) - the scale mechanism here is classic-RL env-worker parallelism, and no published multi-node throughput benchmark was found in the pages read for this card.

**Install**: `pip install DI-engine`; latest PyPI release is 0.5.3, uploaded 2024-12-23, resolving to GitHub tag `v0.5.3` at commit f60b3778cd8f45bd3ea54e317a06ca4f57ace9f5 [9][10][11]. Python `>=3.7` (PyPI metadata; the docs installation page separately states Python 3.7-3.9 and Linux/macOS/Windows, unpinned/live and possibly stale) [9][12]. Apache-2.0 [9]. At the v0.5.3 tag, `install_requires` pins `torch>=1.1.0`, `numpy>=1.18.0,<2`, `gym==0.25.1`, `wandb<=0.19.0`, `flask<=2.0.3`/`werkzeug<=2.0.3`, plus `tensorboardX>=2.2` and about twenty other packages, with per-environment extras (`common_env`, `smac_env`, `mario`, etc.) in `setup.py` [11]. No CUDA/GPU minimum is stated anywhere read for this card; the install docs only say torch will pick up a matching CUDA build automatically if CUDA is already installed [12]. The screening commit (d0b21d0, pushed 2025-12-07, a year ahead of the v0.5.3 release) adds `'transformers'` and `'datasets'` to `install_requires` in `setup.py` - a diff against the v0.5.3 `setup.py` shows this is the only change - meaning `transformers`/`datasets` are NOT dependencies of the package that `pip install DI-engine` currently delivers, only of the unreleased main branch that the GRPO and RLHF-dataset files (also unreleased) live on [13][11].

**Maintained by**: OpenDILab, an org that also maintains sibling projects named in DI-engine's own README (LightZero, DI-star, DI-drive, GenerativeRL) [1]. The last push to the repository's main branch was 2025-12-07 per the screening data, roughly a year after the 2024-12-23 v0.5.3 release, so active development is ahead of the last published package [9].

## Quick start

The repo's canonical task/middleware example, `ding/example/ppo.py`, is a complete PPO-on-CartPole program read in full from the screening commit [2]:

```python
import gym
from ditk import logging
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, \
    gae_estimator, online_logger, ContextExchanger, ModelExchanger
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_ppo_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True, save_cfg=task.router.node_id == 0)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.evaluator_env_num)],
            cfg=cfg.env.manager
        )
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)

        # Consider the case with multiple processes
        if task.router.is_active:
            # You can use labels to distinguish between workers with different roles,
            # here we use node_id to distinguish.
            if task.router.node_id == 0:
                task.add_role(task.role.LEARNER)
            elif task.router.node_id == 1:
                task.add_role(task.role.EVALUATOR)
            else:
                task.add_role(task.role.COLLECTOR)

            # Sync their context and model between each worker.
            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(model))

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(policy.learn_mode, log_freq=50))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(online_logger(train_show_freq=3))
        task.run()


if __name__ == "__main__":
    main()
```

CLI forms, from the same file's own docstring: single-process `python3 -u ding/example/ppo.py`; 4-process (1 learner + 1 evaluator + 2 collectors) `ditask --package . --main ding.example.ppo.main --parallel-workers 4 --topology mesh` [2]. A separate beginner-facing API exists too: `PPOF` in `ding/bonus/ppof.py` wraps the same middleware pipeline behind three lines - `from ding.bonus import PPOF; agent = PPOF(env_id='LunarLander-v2'); agent.train()` - and logs to Weights & Biases by default [14].

## Start it

- One GPU/CPU, one process is the base form shown above; DI-engine reads `cfg.policy.cuda` to decide device placement (`set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)`) rather than defaulting to GPU [2].
- Multiple processes go through the `ditask` CLI, not a separate launcher package: `ditask --package . --main <module>.main --parallel-workers N --topology mesh` starts N workers wired in a mesh so they can exchange context and model state; inside the script, roles are assigned by `task.router.node_id` (`task.add_role(task.role.LEARNER)`, `EVALUATOR`, or `COLLECTOR`) and `ContextExchanger`/`ModelExchanger` middleware sync state between them [2]. There is no bundled multi-GPU sharding launcher (no DeepSpeed/FSDP config templates) in the files read for this card; a separate docs page documents DDP training as its own recipe (`04_best_practice/ddp.html`) [8].
- Configuration is a per-algorithm Python dict (`main_config`/`create_config` in `dizoo/.../config/*.py`), compiled via `compile_config(main_config, create_cfg=create_config, auto=True)`; this is not a base `TrainingArguments`-style class shared across the ecosystem the way trl's Config classes are - each algorithm's config module defines its own fields [2].
- Resuming/warm-starting is a config field, not a CLI flag: `load_ckpt_before_run` in the policy config names a `.pth.tar` path to load before training starts, per the docs' worked CartPole-PPO example [15].
- No GRPO- or generation-specific memory knobs (vLLM offload, `gpu_memory_utilization`, etc.) were found in the files read for this card - `grpo_policy_error` is a bare loss function with no generation-serving layer of its own [4]. No OOM first-aid section was found on the pages read; the general env-parallelism knobs (`collector_env_num`, `evaluator_env_num` in the config) are the obvious levers to reduce for memory, but no library-published OOM guidance was located.

## Watch it

This section covers only the logging mechanics; what a healthy PPO or GRPO curve looks like belongs on those methods' cards.

- **Enable it**: `online_logger(record_train_iter=False, train_show_freq=100)` middleware writes to TensorBoard through a `DistributedWriter`, and raises `RuntimeError("logger writer is None, you should call \`ding_init(cfg)\` at the beginning of training.")` if `ding_init(cfg)` was skipped - so a run with no `ding_init` call produces no TensorBoard log at all [16]. A separate `wandb_online_logger` middleware exists for Weights & Biases and is what the beginner `PPOF.train()` API wires in by default, with `anonymous=True` [16][14].
- **Metric names**: `online_logger`'s scalar keys are read straight from `ctx.train_output`, the dict a policy's `forward()` returns, so the exact metric set is policy-dependent; the two fixed tags it always writes are `basic/eval_episode_return_mean` (evaluation return) and, per training key `k`, `basic/train_<k>` (e.g., a PPO policy's `total_loss`) - both written against `env_step`, or against both `env_step` and `train_iter` if `record_train_iter=True` [16]. This naming scheme is a DI-engine convention read directly from `ding/framework/middleware/functional/logger.py` at the screening commit, not from a docs page.
- **Sample-level logging**: not found in the middleware files read for this card - `online_logger`/`offline_logger` write only scalars (and histograms for keys tagged `[histogram]`); the wandb loggers additionally log action/return distribution plots via internal helper functions, per the same source file, but no example of per-sample generated-text logging was found (unsurprising, since none of the read code wires GRPO/RLHF data through a text-generation loop) [16].
- **Evaluation during training**: built into the pipeline itself via `interaction_evaluator` middleware and an `evaluator_env`, run alongside the collector/trainer middleware in every example read for this card, not an opt-in flag layered on top [2].
- **Stopping**: no RL-specific early-stopping threshold was found in the files read for this card. Configs carry a `stop_value` field (e.g., 195 for CartPole in the docs' load-checkpoint example) that middleware such as `interaction_evaluator` can compare an evaluation return against to end a run, and `PPOF.train()` takes a `termination_checker(max_env_step=step)` middleware capping training by environment steps - both are shapes, not a published numeric threshold or patience recommendation [15][14]. Search covered `ding/framework/middleware/functional/logger.py`, `ding/framework/middleware/ckpt_handler.py`, and the docs' load-checkpoint and config pages; no stopping-rule guidance beyond these config fields was found.

## Save it

- `CkptSaver(policy, save_dir, train_freq=None, save_finish=True)` middleware writes three kinds of files under `<save_dir>/ckpt/`, all via `policy.learn_mode.state_dict()`: `iteration_<train_iter>.pth.tar` every `train_freq` training iterations, `eval.pth.tar` whenever the evaluation return (`ctx.eval_value`) exceeds the best seen so far, and `final.pth.tar` when the task finishes (unless `save_finish=False`) - this is the full contract read from its source, with no separate retention-limit flag that would drop optimizer state [17].
- Each `.pth.tar` is a policy `state_dict`, loaded back by pointing the policy config's `load_ckpt_before_run` field at the file path before starting a run, per the docs' worked example (`cartpole_ppo_config` with `load_ckpt_before_run` set) [15].
- The beginner `PPOF` API layers a `best()` accessor on top: it reads `checkpoint_save_dir/eval.pth.tar` (the best-return checkpoint `CkptSaver` wrote) to reconstruct a ready-to-deploy agent [14].
- No adapter/LoRA saving path exists in DI-engine - `.pth.tar` files are always full policy state dicts, never a partial-adapter format, per every save call read for this card [17][14].
- Whether an external evaluator can load a saved `.pth.tar` directly depends on that evaluator's own loader contract, not on anything DI-engine publishes beyond the state-dict format above - check this skill's shared loading-the-result reference before relying on a save for evaluation.

## Find it in the docs

The docs are the live source; this section is the lookup, not a mirror.

- Address pattern: `https://di-engine-docs.readthedocs.io/en/latest/<chapter>/<page>.html`, with a parallel Chinese tree at `.../zh_CN/latest/...` [1]. The docs are Read-the-Docs "latest" (unpinned, live) and, as fetched 2026-08-11, the installation and load-checkpoint pages both self-report as "DI-engine 0.1.0 documentation" in their page title even though the README's own version banner reads "DI-engine-v0.5.3" - the two numbers disagree, so treat any version number printed on a docs page itself as unreliable and check the README/PyPI page instead [12][15][1].
- Page-slug map, from the docs index's own table of contents (fetched 2026-08-11): `00_intro/index.html` (what DI-engine is), `01_quickstart/installation.html` (install), `01_quickstart/first_rl_program.html` (first RL program), `02_algo/index.html` (algorithm taxonomy by category - offline RL, model-based RL, safe RL, etc.), `03_system/middleware.html` (the task/middleware design), `03_system/config.html` (the config-file system), `03_system/agent.html` (the `PPOF`-style bonus agent API), `04_best_practice/load_ckpt.html` (checkpoint load/resume), `04_best_practice/ddp.html` (DDP distributed training), `05_api_doc/rl_utils.html` (API reference for `ding.rl_utils`, where `grpo_policy_error` lives), `12_policies/index.html` ("RL Algorithms Cheat Sheet", per-algorithm pages such as `12_policies/ppo.html`), `13_envs/index.html` (per-environment tutorial pages) [1].
- The README's own "Algorithm Versatility" table is itself a navigation aid: each of its 57 rows links both a doc page and a `ding/policy/*.py` (or `ding/world_model/*.py`) source path, plus a runnable CLI command such as `ding -m serial_onpolicy -c cartpole_ppo_config.py -s 0` - use it to jump straight from an algorithm name to its source and its runnable demo [1].
- Runnable references beyond the docs: the `dizoo/` companion package ships ready environment configs referenced directly by the quickstart examples (e.g., `dizoo/classic_control/cartpole/config/cartpole_ppo_config.py`, `dizoo/box2d/lunarlander/config/lunarlander_ppo_config.py`) [2][18]. The README also links two Colab notebooks under Quick Start, one specifically a "DI-engine Huggingface Kickoff" notebook - not fetched or verified for this card, so no content claim is made about it beyond its listed title [1].
- Community layer: the README's own curated links point to sibling "awesome" lists maintained by the same OpenDILab org, including `awesome-RLHF` (a curated list of RLHF resources) and `awesome-model-based-RL`, `awesome-decision-transformer`, `awesome-multi-modal-reinforcement-learning`, `awesome-diffusion-model-in-rl` - these are link collections, not DI-engine tutorials themselves, so treat them as pointers one level further out [1]. No practitioner-blog curation page (comparable to trl's `community_tutorials`) was found in the docs table of contents read for this card.
- No official MCP endpoint for querying the docs was found in the pages read for this card.
- Feedback channels named in the README: GitHub issues, a GitHub Discussions forum, a Discord server, a Slack channel, and a contact email (opendilab@pjlab.org.cn) [1]. No maintainer reply in a closed issue was read for this card, so no specific trap-and-issue-number entry is included; the closest documented boundary is the DPO/GRPO gap already stated above under "When to pick it" and "Methods it ships."

## Sources

All GitHub file reads are at the screening commit d0b21d065b2d80317cc16b28a95301be8871d950 unless a release tag is named; all docs/PyPI pages are live/unpinned reads dated 2026-08-11 (docs pages self-report an unrelated "0.1.0" version, noted above and not treated as a pin).

[1] DI-engine README, screening commit d0b21d065b2d80317cc16b28a95301be8871d950. https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/README.md. Fetched 2026-08-11.

[2] `ding/example/ppo.py`, screening commit. https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/ding/example/ppo.py. Fetched 2026-08-11.

[3] DI-engine GitHub repository. https://github.com/opendilab/DI-engine. Fetched 2026-08-11.

[4] `ding/rl_utils/grpo.py`, screening commit (the shortlist row's GRPO citation). https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/ding/rl_utils/grpo.py. Fetched 2026-08-11. Also confirmed added to the repository around 2025-03-01, via the file's commit history page: https://github.com/opendilab/DI-engine/commits/main/ding/rl_utils/grpo.py, fetched 2026-08-11.

[5] `ding/utils/data/rlhf_offline_dataset.py`, screening commit (the shortlist row's RLHF citation). https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/ding/utils/data/rlhf_offline_dataset.py. Fetched 2026-08-11.

[6] Confirmed absence of `ding/policy/dpo.py` at the screening commit via a direct HTTP 404: https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/ding/policy/dpo.py. Checked 2026-08-11.

[7] `ding/framework/middleware/functional/trainer.py`, screening commit (the shortlist row's Trainer citation). https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/ding/framework/middleware/functional/trainer.py. Fetched 2026-08-11.

[8] DI-engine docs, "Using DDP Distributed Training in DI-engine". https://di-engine-docs.readthedocs.io/en/latest/04_best_practice/ddp.html. Page existence and title confirmed via the docs index table of contents fetched 2026-08-11; https://di-engine-docs.readthedocs.io/en/latest/index.html.

[9] DI-engine on PyPI (version, release date, author, license, requires_python). https://pypi.org/pypi/DI-engine/json. Fetched 2026-08-11.

[10] DI-engine GitHub Releases list (confirms v0.5.3 is the newest tag). https://github.com/opendilab/DI-engine/releases. Fetched 2026-08-11.

[11] `setup.py` at the v0.5.3 tag (dependency floors/caps and extras for the released package). https://raw.githubusercontent.com/opendilab/DI-engine/v0.5.3/setup.py. Fetched 2026-08-11. The v0.5.3 tag's commit SHA, f60b3778cd8f45bd3ea54e317a06ca4f57ace9f5, was read from https://github.com/opendilab/DI-engine/releases/tag/v0.5.3, fetched 2026-08-11.

[12] DI-engine docs, Installation Guide. https://di-engine-docs.readthedocs.io/en/latest/01_quickstart/installation.html. Fetched 2026-08-11.

[13] `setup.py` at the screening commit, diffed against [11]; the only difference is two added `install_requires` entries, `'transformers'` and `'datasets'`. https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/setup.py. Fetched 2026-08-11.

[14] `ding/bonus/ppof.py`, screening commit (the `PPOF` beginner agent API: `train`, `deploy`, `best`, default wandb logging, `termination_checker`, `checkpoint_save_dir`). https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/ding/bonus/ppof.py. Fetched 2026-08-11.

[15] DI-engine docs, "Loading Pre-trained Models and Resuming Training" (the `load_ckpt_before_run` config field and its worked CartPole-PPO example, including the `stop_value` field). https://di-engine-docs.readthedocs.io/en/latest/04_best_practice/load_ckpt.html. Fetched 2026-08-11.

[16] `ding/framework/middleware/functional/logger.py`, screening commit (`online_logger`, `offline_logger`, `wandb_online_logger`, `wandb_offline_logger` and their TensorBoard/W&B calls). https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/ding/framework/middleware/functional/logger.py. Fetched 2026-08-11.

[17] `ding/framework/middleware/ckpt_handler.py`, screening commit (the `CkptSaver` class and its full save contract). https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/ding/framework/middleware/ckpt_handler.py. Fetched 2026-08-11.

[18] `ding/example/ppo_lunarlander.py`, screening commit (a second complete quickstart-style example, cited to show the config path pattern). https://raw.githubusercontent.com/opendilab/DI-engine/d0b21d065b2d80317cc16b28a95301be8871d950/ding/example/ppo_lunarlander.py. Fetched 2026-08-11.

Ecosystem tools named only in passing (Weights & Biases, TensorBoard, Discord, Slack) are reached through the sources above and are not separately enumerated. LightZero, DI-star, DI-drive, GenerativeRL and the other OpenDILab sibling projects named in [1] are not cited beyond that mention.
