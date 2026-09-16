# FlashAttention 2

Read in step 7 when attention memory or speed binds.

It appears widely in public training-run records:

```text
attn_implementation="flash_attention_2"
```

The main gains:

- Less memory for the intermediate results inside attention.
- Higher throughput on long sequences.
- Lower cost for long-context SFT.
- A larger micro-batch becomes possible.

Whether it applies depends on:

- Architecture support.
- The dtype.
- The CUDA, PyTorch, and flash-attn versions.
- The form of the attention mask.
- Whether there is sliding-window or other special attention.

It does not solve:

- An OOM caused by vocabulary logits that are too large.
- Optimizer states.
- MLP activations that are too large.
- Running out of disk when saving the model.

QLoRA runs and full fine-tuning runs alike set FlashAttention 2 explicitly rather than trusting a default. Turn it on unless something real keeps it off: leave it out only when it will not install after real attempts -- build settings, a matching prebuilt wheel, a nearby version -- or this architecture cannot use it, and write that reason down.

The framework skill's library cards show where each trainer turns FlashAttention on; this card keeps only what the workflow needs.
