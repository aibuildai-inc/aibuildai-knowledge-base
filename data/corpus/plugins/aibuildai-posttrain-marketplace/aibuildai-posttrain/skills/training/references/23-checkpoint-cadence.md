# Checkpoint cadence and `load_best_model_at_end`

Read in step 8 when setting save and eval intervals.

Many Trainer scripts use:

```text
eval_strategy="steps"
save_strategy="steps"
load_best_model_at_end=True
metric_for_best_model="eval_loss"
```

This stops you from taking only the last checkpoint, but it has one important limit:

> **The lowest validation loss is not always the highest benchmark score.**

Reasons include:

- the validation source differs from the benchmark;
- the benchmark depends more on exact format;
- decoding has nothing to do with loss;
- a later checkpoint has slightly worse loss but better stopping behavior;
- the reward curve of an RL checkpoint swings;
- a small benchmark has larger variance.

So `load_best_model_at_end` works as a first screen. It must not be the only rule for the final choice.
