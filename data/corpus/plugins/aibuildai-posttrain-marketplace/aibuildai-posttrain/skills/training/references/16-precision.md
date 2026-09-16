# Precision and quantized training

Read in steps 7 and 13: training precision and delivered precision.

## BF16

BF16 is the most common training precision in public training-run records. You can use it for:

- Full fine-tuning.
- LoRA.
- The compute dtype of QLoRA.
- Evaluation.

But BF16 is not a lossless conversion.

A GRPO checkpoint can do better in FP32: after the conversion to BF16, the same evaluation can lose a few items, in which case the run submits the FP32 checkpoint.

This shows the final dtype is also an artifact decision you must verify.

## BitsAndBytes 4-bit

Common for QLoRA:

```text
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=True
bnb_4bit_compute_dtype=bfloat16
```

In this setup:

- NF4 is the 4-bit weight representation.
- Double quantization compresses the quantization constants further.
- The forward and backward compute can still run in BF16.
- The base weights are usually frozen.
- What you train is the LoRA parameters.

## The 8-bit optimizer

`paged_adamw_8bit` compresses the optimizer states and handles peaks with paged memory. It and 4-bit model loading are two different things.

The framework skill's library cards show where each trainer wires BitsAndBytes in, and its `references/loading-the-result.md` covers what the delivered folder must hold; this card keeps only what the workflow needs.
