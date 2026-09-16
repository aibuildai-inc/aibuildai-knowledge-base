# Gradient checkpointing, `use_cache=False`, and the micro-batch

Read in step 7 when memory runs out.

Gradient checkpointing does not keep every forward activation. It recomputes them during the backward pass.

The typical combination:

```text
model.config.use_cache = False
gradient_checkpointing=True
use_reentrant=False
```

Good when:

- You do full fine-tuning.
- The context is long.
- The vocabulary is large.
- You want a bigger batch on QLoRA.
- The policy forward pass in GRPO uses a lot of memory.

The cost:

- Training gets slower.
- Some architectures do not work with reentrant checkpointing.
- You must check `use_cache`.
- The order of initialization with quantization and PEFT can matter.

When memory runs out, the common order of adjustment in the records is:

1. Lower the micro-batch.
2. Raise gradient accumulation.
3. Turn on gradient checkpointing.
4. Shorten the max sequence.
5. Use packing or group by length.
6. Move to QLoRA.
7. Cut the LoRA targets or rank.
8. Change the optimizer.
9. Only then think about cutting the data or the model.

A run that hits an OOM on large-vocab logits changes the batch size; a QLoRA run can use non-reentrant gradient checkpointing.

The methodology skill's card on GRPO carries the full method, and the framework skill's cards carry the library detail; this card keeps only what the workflow needs.
