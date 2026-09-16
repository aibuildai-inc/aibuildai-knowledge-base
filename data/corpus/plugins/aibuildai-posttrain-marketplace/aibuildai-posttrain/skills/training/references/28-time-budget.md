# Time budget and parallel experiment planning

Read in steps 7 and 8 when splitting the clock, and whenever the GPU idles.

Under a tight wall-clock budget, planning the clock is itself a trick.

Common moves:

- prepare the next version of the data on the CPU while the GPU trains;
- write the training script while the GPU generates data;
- average checkpoints on the CPU;
- run training in the background and read the evaluations you already have in the foreground;
- run a pilot first to learn tokens per second and the OOM point;
- screen on a fast subset, and spend a full evaluation only on strong candidates;
- always keep a verified incumbent;
- reserve time for packaging and the smoke test before the end;
- use `save_total_limit` to control disk;
- delete useless checkpoints and kill runaway processes.

A run can prepare a soup during its stage-2 training and evaluate the main candidate at the same time. A run that already holds a verified incumbent can treat a new version as a pure gain experiment: it replaces the incumbent only if a full evaluation clearly wins.

### A suggested shape for the window

| Phase | Main goal |
|---|---|
| first 5-10% | evaluator contract, baseline, what data is available |
| 10-20% | pilot SFT, throughput and OOM check |
| 20-65% | main training |
| 65-85% | second stage / RFT / RL / checkpoint evaluation |
| 85-95% | decoding, repeated evaluation, soup |
| last 5-15% | merge, artifact verification, final smoke test |

These are not fixed ratios, but a run must reserve a finalization buffer on purpose.
