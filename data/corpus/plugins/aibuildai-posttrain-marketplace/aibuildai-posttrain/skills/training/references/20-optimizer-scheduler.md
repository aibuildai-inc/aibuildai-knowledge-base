# Optimizer, scheduler, and stability settings

Read in step 7 when setting the schedule.

The most common choice in the records is the AdamW family:

- `adamw_torch`
- `adamw_torch_fused`
- `paged_adamw_8bit`

Common schedulers:

- Cosine.
- A few linear.
- A warmup ratio near 0.03 appears in many examples.

Common stability settings:

- `max_grad_norm=0.3` or `1.0`.
- Weight decay.
- A seed and a data seed.
- BF16.
- Non-reentrant gradient checkpointing.
- Frequent logging.
- Eval and save every N steps.

## Observed magnitudes, not fixed defaults

| Case | Common magnitude in the records |
|---|---:|
| Full FT SFT | around `1e-5` |
| LoRA / QLoRA SFT | `1e-4` to `2e-4` |
| Second-stage anneal | lower than the main SFT |
| GRPO | `5e-7` to `5e-6` |
| Warmup ratio | about `0.03` |
| LoRA dropout | `0` to `0.1` |

For example, a QLoRA run might use `2e-4`, cosine, and 3% warmup; a full fine-tuning run `1e-5`; a GRPO run a much lower learning rate near `5e-7`.

The point is not to copy the numbers. Set the magnitude from:

- Full fine-tuning or an adapter.
- The number of trainable parameters.
- The amount of data.
- The number of epochs.
- Whether you continue from an already trained checkpoint.
- How sensitive the benchmark is to forgetting.

The methodology skill's card on GRPO carries the full method; this card keeps only what the workflow needs.
