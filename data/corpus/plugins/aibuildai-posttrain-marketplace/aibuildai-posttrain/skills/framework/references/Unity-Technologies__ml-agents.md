# ml-agents

Unity's game-engine-native RL/imitation-learning toolkit: instrument a Unity scene as an environment in C#, train the policy from a Python side-process with `mlagents-learn`, and run inference back inside Unity through the Inference Engine.

**ML-Agents** (the Unity Machine Learning Agents Toolkit) is "an open-source project that enables games and simulations to serve as environments for training intelligent agents using deep reinforcement learning and imitation learning" [1]. It is built and maintained by Unity Technologies [2][3]. The project ships as three packages with one shared API: a C# Unity package, `com.unity.ml-agents`, that you attach to GameObjects in a scene to define Agents, observations, and actions; and two Python packages, `mlagents` (the trainers, exposed through the `mlagents-learn` CLI) [5] and `mlagents_envs` (the Python-side low-level API, which communicates with the Learning Environment through an "External Communicator" that lives inside it) [7]. It lives at https://github.com/Unity-Technologies/ml-agents [3].

**When to pick it**: you have (or are building) a Unity scene and want to train agents that act inside that engine's physics and rendering, not a text/LLM post-training target - this is the one card in this deck built around a game engine rather than a language model, so it is not a substitute for a text-based RLHF library such as trl or verl. Training runs on your own machine (Editor or Standalone build) driving a separate Python trainer process through the External Communicator [7]; there is no cluster launcher or multi-GPU story (see Scale it handles). Pick trl/verl instead if the target is a Hugging Face-style causal LM; pick ml-agents if the target is a Unity-simulated agent, including multi-agent competitive or cooperative games.

**Methods it ships** [7]: two RL algorithms, PPO (the default) and SAC (off-policy, more sample-efficient but update-heavier, recommended for slower environments) [7]; two sparse-reward intrinsic-reward modules, Curiosity (an inverse/forward-model pair, `curiosity` reward signal) and RND (Random Network Distillation, a fixed-random-network/predictor pair) [7]; two imitation-learning methods that consume recorded demonstrations, GAIL (Generative Adversarial Imitation Learning, an adversarial discriminator reward) and Behavioral Cloning (BC, enabled as a setting on the PPO or SAC trainer rather than its own trainer) [7]; and two environment-specific multi-agent trainers, Self-Play (symmetric or asymmetric adversarial training against fixed past checkpoints of the opponent, scored with an ELO rating, implemented in `mlagents.trainers.ghost.trainer`) [7][8][9] and MA-POCA (MultiAgent POsthumous Credit Assignment, a centralized-critic trainer for cooperative groups that keeps crediting agents removed mid-episode) [7]. Curriculum Learning and Environment Parameter Randomization are environment-configuration mechanisms, not trainers, layered on top of any of the above [7]. None of these are flagged experimental on the current docs page [7]; the taxonomy lives at the live overview page, recheck there [7].

**Scale it handles**: single machine only, documented as such - there is no distributed/multi-node launcher, and `torch_settings: device` takes a single device (e.g. `cpu`) with no multi-GPU field [10]. A maintainer confirmed multi-GPU training was unsupported and only on the roadmap as of 2022 (see the trap in Find it in the docs) [30]. Parallelism is `--num-envs=<n>`, which spawns concurrent Unity instances feeding one training process on one machine, capped in practice by local CPU/GPU/RAM, and the docs warn that changing `--num-envs` with everything else fixed changes the learned result [10]. No published benchmark backs a specific `--num-envs` ceiling; it is documented as a resource-constrained knob, not a measured scaling curve [10].

**Install**: two separate packages, two separate version tracks, and they are currently out of step. The Python trainer is `python -m pip install mlagents==1.1.0`, released 2024-10-05, requiring Python `>=3.10.1,<=3.10.12`, Apache-2.0 licensed [12][13][14]; its pinned floors at that release are `torch>=2.1.1`, `numpy>=1.23.5,<1.24.0`, `onnx==1.15.0`, `protobuf>=3.6,<3.21`, `grpcio>=1.11.0,<=1.48.2`, `tensorboard>=2.14`, and `mlagents_envs==1.1.0` [14]; the release's own requirements file is `ml-agents/setup.py` at the `release_22` tag (commit `200fe54e14b649d6eac66a7f0779c1086c506919`) [14]. The `mlagents_envs==1.1.0` dependency it pulls in has its own requirements file, `ml-agents-envs/setup.py`, at that same tag: it pins the old `gym>=0.21.0` API, not `gymnasium`, plus `pettingzoo==1.15.0` exactly, both of which can collide with newer RL-ecosystem packages a reader already has installed [6]. That release pairs with the C# Unity package version 3.0.0 (Unity Editor floor 2023.2, PyTorch 2.1.1) per the matching GitHub release notes [15]. The screening commit `ab179e18df7197d644f08637d8acf1fc4a1d5014` sits on the default `develop` branch, ahead of every tagged release: its `com.unity.ml-agents/package.json` already reads version 4.1.0 with a Unity floor of 6000.0 [16], and the newest tagged GitHub release, `release_23_tag` (C# package 4.0.0, published 2025-09-02, Unity floor bumped to 6000.0, `grpcio` bumped to `<=1.53.2`), has not been mirrored to PyPI - `pip install mlagents` today still delivers 1.1.0 [15][17][12]. The Unity C# package is installed separately, by name (`com.unity.ml-agents`) through the Unity Package Manager, which pulls from Unity's package registry rather than a pip-style pinned file [18]. No CUDA/hardware minimum is stated for the Python package beyond the torch floor; the installation docs mention a Windows-specific `pip3 install torch~=2.2.1 --index-url .../cu121` step for CUDA support [18].

**Maintained by**: Unity Technologies [2][3]; about 19,615 GitHub stars (not a ranking signal) [2]; actively developed - the repository's last push was 2026-08-07, and a `release/4.1.0` branch (commit `ee0a08ccae597094003844d0121317f9790a1676`) is already in progress toward the next tagged release beyond 4.0.0/1.1.0 [2][19].

## Quick start

The full 3D Balance Ball walkthrough, from the docs' own sample workflow, run from the cloned repo root [20]:

```sh
mlagents-learn config/ppo/3DBall.yaml --run-id=first3DBallRun
```

then press Play in the Unity Editor with the `3DBall` scene open (`Assets/ML-Agents/Examples/3DBall/Scenes`) when prompted [20]. To resume that same run later:

```sh
mlagents-learn config/ppo/3DBall.yaml --run-id=first3DBallRun --resume
```
[20]

The general form, for any config and any Unity executable or Editor session [21]:

```sh
mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>
```

The docs publish the console log of that exact 3DBall run as evidence it works: `Mean Reward` climbs from 1.242 at Step 1000 (Std of Reward 0.746) to 27.284 at Step 10000 (Std of Reward 28.667), and the page states that "If training is succeeding, the `Mean Reward` value printed to the screen increases as training progresses" [20].

## Start it

- One machine is the only supported shape: run the Unity Editor or a Standalone build (or omit `--env` to train against the open Editor scene) alongside one `mlagents-learn` process [21].
- Parallel Unity instances on that one machine: `--num-envs=<n>` (optionally with `--base-port`) spawns `n` concurrent Unity environment instances feeding the same trainer process; the docs flag this as resource-constrained by the local machine and note that changing `n` changes the training result even with hyperparameters held fixed [10].
- Config surface: CLI arguments can also be embedded in the training YAML, grouped into `env_settings` (env path, `num_envs`, ports, seed), `engine_settings` (rendering `width`/`height`, `time_scale`, `no_graphics` for headless runs), `checkpoint_settings` (`run_id`, `resume`, `force`, `initialize_from`, `inference`), and `torch_settings` (`device`) [10]. The library's own example config sets `torch_settings: device: cpu`, i.e. it does not default to a GPU device for you - set `device` explicitly for GPU training [10].
- Effective sample throughput is governed by PPO/SAC's `buffer_size` (experiences collected before an update; default 10240 for PPO, 50000 for SAC) versus `batch_size` (experiences per gradient step, meant to be several times smaller than `buffer_size`) [22].
- Headless/remote runs need `--no-graphics` (or `no_graphics: true`); the FAQ documents this as the fix for a `Communicator was unable to connect` timeout when running on a server with no display, alongside removing any `HTTP_PROXY`/`HTTPS_PROXY` environment variables that can also cause the same timeout [23].
- Out-of-memory / instability first aid found in the docs read for this card: reduce `--num-envs`, reduce `batch_size`/`buffer_size`, or reduce `engine_settings.width`/`height` and `time_scale`; a `Mean reward : nan` under PPO is diagnosed in the FAQ as episodes never terminating, fixed by setting a Agent `Max Steps` > 0 in the Scene Inspector [23][10].

## Watch it

This section covers the mechanics only; what a metric shape means for a given method is not restated here.

- Logging is on by default with no separate tracker to enable: every `mlagents-learn` run writes TensorBoard summaries under `results/<run-identifier>` with no extra flag, viewed with `tensorboard --logdir results --port 6006` [24].
- The published per-category metric names, from the TensorBoard guide [24]: Environment - `Environment/Lesson`, `Environment/Cumulative Reward`, `Environment/Episode Length`; `Is Training`; Policy - `Policy/Entropy`, `Policy/Learning Rate` (PPO, SAC), `Policy/Entropy Coefficient` (SAC), `Policy/Extrinsic Reward`, `Policy/Value Estimate` (PPO, SAC), and the Curiosity/GAIL pairs `Policy/Curiosity Reward`, `Policy/Curiosity Value Estimate`, `Policy/GAIL Reward`, `Policy/GAIL Value Estimate`, `Policy/GAIL Policy Estimate`, `Policy/GAIL Expert Estimate`; Losses - `Losses/Policy Loss`, `Losses/Value Loss` (PPO, SAC), `Losses/Forward Loss`, `Losses/Inverse Loss` (Curiosity), `Losses/Pretraining Loss` (BC), `Losses/GAIL Loss`; Self-Play - `Self-Play/ELO` [24].
- Custom Unity-side metrics can be pushed into the same TensorBoard run from C# with `Academy.Instance.StatsRecorder.Add("MyMetric", 1.0)` [24].
- No sample-level generation logging exists as a library feature - there is nothing analogous to a chat-completion log here, since the "generations" are Unity episode rollouts, not text; per-episode reward and length are the `Environment/*` scalars above [24].
- Evaluation-during-training is not a separate documented config surface distinct from ordinary training monitoring in the pages read for this card; the closest built-in signal is the same `Environment/Cumulative Reward` scalar tracked live.
- No published stopping-rule threshold was found: none of the Training-ML-Agents guide, the Training-Configuration-File reference, or the TensorBoard guide state a target reward, patience, or auto-stop criterion - training runs until `max_steps` or a manual `Ctrl+C` [21][22][24]. A run interrupted with `Ctrl+C` saves its current checkpoint before exiting [21].

## Save it

- Artifacts land under `results/<run-identifier>/`: TensorBoard summaries, model checkpoints, a final model file, and a `run_logs` folder with a timers file [21].
- Two paired files per checkpoint, named `<behavior_name>-<step>.pt` and `<behavior_name>-<step>.onnx`, written by `TorchModelSaver.save_checkpoint`: the `.pt` file holds a `state_dict` for every registered module (policy plus optimizer/optimizer-adjacent modules) via plain `torch.save`, and a copy of that same state is written to a fixed `checkpoint.pt` in the run directory, which is what `--resume` loads [25]. The `.onnx` file is the exported inference-only model produced by the same call [25]. Checkpoint cadence and retention are `checkpoint_interval` (default 500000 experiences per the config reference, 50000 in the docs' own sample config) and `keep_checkpoints` (default 5, an oldest-first rotation count) - both are top-level trainer settings nested directly under each behavior's config block (`behaviors.<name>.checkpoint_interval`), not under `checkpoint_settings`, which holds only `run_id`/`initialize_from`/`load_model`/`resume`/`force`/`train_model`/`inference` [22][10]. There is no documented flag to drop optimizer state from an individual checkpoint - every `.pt` written always carries the full module state [25].
- The single final model is written as `results/<run-identifier>/<behavior_name>.onnx` once training completes or is interrupted; this is the file used for Sentis-based inference back in Unity [21][11].
- Resume: `mlagents-learn <config> --run-id=<id> --resume` loads `checkpoint.pt` from that run's results directory and continues, including global step count [21][25]. `--force` re-runs the same `run-id` and overwrites prior artifacts instead of resuming [21]. `--initialize-from=<old-run-id>` starts a new run seeded from another run's weights without inheriting its step count; if the network architecture changed (e.g. a new reward signal, or changed `hidden_units`), the loader keeps the parts that still match by shape and re-initializes the rest, logging which keys were missing or unexpected [21][25].
- No adapter/LoRA-style partial-save concept exists here - PPO/SAC/GAIL/BC train the full policy network, and the `.pt`/`.onnx` pair together is the complete trained model, not a delta on a frozen base.
- Loader handoff: the `.onnx` file is the artifact an evaluator or the Unity Inference Engine (Sentis) loads directly for inference, dropping it into a Behavior's `Model` field; the `.pt` file is for resuming training in Python only, not for Unity-side inference [11][25].

## Find it in the docs

The current authoritative docs are the in-repo `com.unity.ml-agents/Documentation~/*.md` pages, also published live at `https://docs.unity3d.com/Packages/com.unity.ml-agents@latest`; the old top-level `docs/` folder in the repository is explicitly deprecated and repoints here [26]. The repository's root README carries its own, separate migration notice, pointing a reader away from the old web docs at `unity-technologies.github.io/ml-agents` toward this same Unity Package documentation [4]. `@latest` 301-redirects to a version-pinned URL (confirmed by fetch, 2026-08-10); a version can be pinned directly as `https://docs.unity3d.com/Packages/com.unity.ml-agents@4.1/manual/index.html`, which also loads [26].

- Page-slug recipe for the in-repo source: `com.unity.ml-agents/Documentation~/<Page-Name>.md`, e.g. `Training-ML-Agents.md`, `Training-Configuration-File.md`, `Using-Tensorboard.md`, `ML-Agents-Overview.md`, `Installation.md`, `FAQ.md`, `Package-Limitations.md`, `Hugging-Face-Integration.md`, `Sample.md`, `Python-Gym-API.md`, `Python-APIs.md`, `Tutorial-Colab.md`, `Migrating.md` [26].
- Question-to-page map: how to install -> `Installation.md` [18]; how to start/resume/configure a run -> `Training-ML-Agents.md` [21]; every hyperparameter and its default/typical range -> `Training-Configuration-File.md` [22]; what each TensorBoard scalar means -> `Using-Tensorboard.md` [24]; method definitions (PPO/SAC/Curiosity/RND/GAIL/BC/Self-Play/MA-POCA/Curriculum/Environment Parameter Randomization) -> `ML-Agents-Overview.md` [7]; known constraints -> `Package-Limitations.md` (C# package) plus the linked `README.md#limitations` of the `mlagents` and `mlagents_envs` Python packages [27][28].
- Runnable references beyond the docs: the `config/` directory (read at the screening commit) ships ready hyperparameter YAMLs per method, including `config/ppo/3DBall.yaml`, `config/imitation/PushBlock.yaml`, and `config/imitation/Crawler.yaml` [31]; the example Unity scenes referenced throughout the docs (3D Balance Ball, Pyramids, PushBlock, Tennis, Soccer, Wall Jump) are the known-good smoke tests, opened from `Assets/ML-Agents/Examples/*` after following `Examples-setup.md` [20][7].
- Hugging Face integration is a first-party feature, not a third-party bridge: `mlagents-push-to-hf --run-id=<id> --local-dir=<results dir> --repo-id=<user>/<repo> --commit-message=<msg>` uploads a trained run, and `mlagents-load-from-hf --repo-id=<user>/<repo> --local-dir=<dir>` downloads one; community ML-Agents models are browsable at `huggingface.co/models?library=ml-agents` [29].
- Community layer: the docs' own Hugging Face page curates two tutorials - a short Hugging Face Deep RL Course unit on training "Huggy the Dog" and a longer in-depth unit - both hosted at `huggingface.co/learn/deep-rl-course` [29]; no separate official ml-agents blog/tutorial index beyond that page and the in-repo docs was found in the pages read for this card.
- Trap found in a closed issue: in issue #5720, "Multiple GPU support" (closed 2022-03-29), a Unity contributor replied "We do have plans to improve training performance on our roadmap for end of 2022/beginning of 2023. This would include adding support for, among many things, multi-GPU training" (miguelalonsojr, author association CONTRIBUTOR, 2022-03-29) [30]. A separate issue, #5644 "Will support for multi-gpu return again?", was also closed (2023-10-08) with no maintainer follow-up in the same thread [30]. Multi-GPU training is not documented as supported anywhere in the docs read for this card, consistent with the single-`device` `torch_settings` field [10].
- Honest boundary: training only runs in the Unity Editor or a Standalone build with the Mono scripting backend - IL2CPP is not supported for training, and an environment falls back to inference mode if training isn't currently supported or running; enabling Headless mode disables visual observations; physics/inference speed is capped at 100x real-time [27].

## Sources

Ecosystem tools named in passing (PyTorch, ONNX, TensorBoard, Sentis/the Unity Inference Engine, gRPC) are reached through the docs pages cited below and are not separately enumerated. Method names (PPO, SAC, GAIL, RND, MA-POCA, Curriculum Learning) are cited to the toolkit's own overview page, not their defining papers, per this deck's convention. All docs pages are read at the versions/dates stated; source-code claims carry the commit read.

[1] ML-Agents repository description (GitHub API, `repos/Unity-Technologies/ml-agents`). https://github.com/Unity-Technologies/ml-agents. Fetched 2026-08-10.

[2] Unity-Technologies/ml-agents GitHub API repository metadata (stars, pushed_at, license, description). https://api.github.com/repos/Unity-Technologies/ml-agents. Fetched 2026-08-10.

[3] ml-agents GitHub repository. https://github.com/Unity-Technologies/ml-agents. Fetched 2026-08-10.

[4] ml-agents Readme.md (repository root: project description, feature list, and its own Documentation Migration Notice pointing from the deprecated web docs to the Unity Package documentation). https://github.com/Unity-Technologies/ml-agents/blob/develop/Readme.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[5] ml-agents/README.md (Python trainers package description). https://github.com/Unity-Technologies/ml-agents/blob/release_22/ml-agents/README.md, read at the release_22 tag (commit 200fe54e14b649d6eac66a7f0779c1086c506919). Fetched 2026-08-10.

[6] ml-agents-envs/setup.py at the release_22 tag (install_requires pins for the mlagents_envs package actually shipped by `pip install mlagents`, including `gym>=0.21.0` and `pettingzoo==1.15.0`). https://raw.githubusercontent.com/Unity-Technologies/ml-agents/release_22/ml-agents-envs/setup.py, resolved from tag release_22 to commit 200fe54e14b649d6eac66a7f0779c1086c506919. Fetched 2026-08-10.

[7] ML-Agents Theory / overview doc (method list and descriptions: PPO, SAC, Curiosity, RND, GAIL, BC, Self-Play, MA-POCA, Curriculum Learning, Environment Parameter Randomization). com.unity.ml-agents/Documentation~/ML-Agents-Overview.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014; live at https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/manual/ML-Agents-Overview.html. Fetched 2026-08-10.

[8] ghost/trainer.py (Self-Play trainer implementation, referenced in the shortlist row's methods_seen). https://github.com/Unity-Technologies/ml-agents/blob/develop/ml-agents/mlagents/trainers/ghost/trainer.py, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[9] ML-Agents Overview doc, Self-Play section (ELO rating, symmetric/asymmetric games). Same source as [7].

[10] Training-ML-Agents.md, CLI/config-groups section (env_settings, engine_settings, checkpoint_settings, torch_settings; --num-envs behavior and warning). com.unity.ml-agents/Documentation~/Training-ML-Agents.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[11] Sample.md, inference-workflow section (Sentis/Inference Engine, CPU/GPU inference device, dragging the `.onnx` file into Behavior Parameters). com.unity.ml-agents/Documentation~/Sample.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[12] mlagents on PyPI (current version, upload date). https://pypi.org/pypi/mlagents/json. Fetched 2026-08-10.

[13] mlagents PyPI metadata, requires_python field. Same source as [12].

[14] ml-agents/setup.py at the release_22 tag (install_requires pins, python_requires, entry_points). https://raw.githubusercontent.com/Unity-Technologies/ml-agents/release_22/ml-agents/setup.py, resolved from tag release_22 to commit 200fe54e14b649d6eac66a7f0779c1086c506919. Fetched 2026-08-10.

[15] GitHub Releases list for Unity-Technologies/ml-agents (release_22 and release_23_tag release notes: C# package versions, Unity floor, PyTorch/grpcio/numpy/onnx/protobuf bumps). https://api.github.com/repos/Unity-Technologies/ml-agents/releases. Fetched 2026-08-10.

[16] com.unity.ml-agents/package.json at the screening commit (in-development version 4.1.0, Unity floor 6000.0). https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/package.json, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[17] com.unity.ml-agents/package.json at the release_22 tag (C# package version 3.0.0, Unity floor 2023.2). https://raw.githubusercontent.com/Unity-Technologies/ml-agents/release_22/com.unity.ml-agents/package.json, resolved from tag release_22 to commit 200fe54e14b649d6eac66a7f0779c1086c506919. Fetched 2026-08-10.

[18] Installation.md (Unity Editor floor, Package Manager install-by-name, Conda/Python install steps, Windows CUDA torch install line). com.unity.ml-agents/Documentation~/Installation.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[19] GitHub API branch lookup for release/4.1.0 (in-progress next release branch, commit ee0a08ccae597094003844d0121317f9790a1676). https://api.github.com/repos/Unity-Technologies/ml-agents/branches/release%2F4.1.0. Fetched 2026-08-10.

[20] Sample.md, "Training a New Model with Reinforcement Learning" section (mlagents-learn quickstart command, --resume example). com.unity.ml-agents/Documentation~/Sample.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[21] Training-ML-Agents.md, "Training with mlagents-learn" section (basic command form, artifacts, --resume, --force, --initialize-from, partial-model-load behavior). Same source as [10].

[22] Training-Configuration-File.md (batch_size/buffer_size defaults and typical ranges, checkpoint_interval/keep_checkpoints defaults in the sample config, BC/GAIL hyperparameter defaults). com.unity.ml-agents/Documentation~/Training-Configuration-File.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[23] FAQ.md (headless/--no-graphics fix for connection timeout, HTTP_PROXY cause, Mean reward: nan cause and fix). com.unity.ml-agents/Documentation~/FAQ.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[24] Using-Tensorboard.md (how to launch TensorBoard, full per-category metric name list, StatsRecorder custom metrics). com.unity.ml-agents/Documentation~/Using-Tensorboard.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[25] mlagents/trainers/model_saver/torch_model_saver.py (checkpoint file writing, state_dict contents, resume/initialize-from loading logic and key-mismatch warnings). https://github.com/Unity-Technologies/ml-agents/blob/develop/ml-agents/mlagents/trainers/model_saver/torch_model_saver.py, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[26] docs/Readme.md and docs/index.md (deprecation notice pointing to com.unity.ml-agents/Documentation~ and the live docs.unity3d.com page), plus direct fetch of https://docs.unity3d.com/Packages/com.unity.ml-agents@latest (301 redirect) and https://docs.unity3d.com/Packages/com.unity.ml-agents@4.1/manual/index.html (loads). docs/Readme.md and docs/index.md read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[27] Package-Limitations.md (Mono-only training, IL2CPP unsupported, Headless mode disables visual observations, 100x real-time physics cap). com.unity.ml-agents/Documentation~/Package-Limitations.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[28] Limitations.md (index page linking to the mlagents and mlagents_envs Python package README limitations sections). com.unity.ml-agents/Documentation~/Limitations.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[29] Hugging-Face-Integration.md (mlagents-push-to-hf / mlagents-load-from-hf usage, curated tutorial links). com.unity.ml-agents/Documentation~/Hugging-Face-Integration.md, read at commit ab179e18df7197d644f08637d8acf1fc4a1d5014. Fetched 2026-08-10.

[30] GitHub Issues search for multi-GPU (issues #5644 "Will support for multi-gpu return again?" and #5720 "Multiple GPU support"), plus the comment thread on #5720 (miguelalonsojr, CONTRIBUTOR, 2022-03-29). https://api.github.com/search/issues?q=repo:Unity-Technologies/ml-agents+multi-gpu and https://api.github.com/repos/Unity-Technologies/ml-agents/issues/5720/comments. Fetched 2026-08-10.

[31] Full recursive git tree at the screening commit (verifies config/ file paths, including config/imitation/Crawler.yaml and config/imitation/PushBlock.yaml; config/imitation/CrawlerStatic.yaml does not exist despite appearing in [7]). https://api.github.com/repos/Unity-Technologies/ml-agents/git/trees/ab179e18df7197d644f08637d8acf1fc4a1d5014?recursive=1. Fetched 2026-08-10.
