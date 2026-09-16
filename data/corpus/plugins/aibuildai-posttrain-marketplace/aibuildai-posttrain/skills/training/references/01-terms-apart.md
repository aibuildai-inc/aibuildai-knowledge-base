# Terms that are easy to mix up

Background for steps 7 and 11: terms that are easy to mix up.

Several names get used as if they meant the same thing. They do not. This table keeps them apart.

| Name | What it really is | What it is not |
|---|---|---|
| **BF16** | A 16-bit floating point precision for training and inference | Not quantization in the BitsAndBytes sense |
| **QLoRA** | Stores the frozen base weights in 4-bit and trains only the LoRA adapter; compute is usually still BF16 | Not a direct update of 4-bit base weights |
| **8-bit AdamW** | Compression of the optimizer state | Not the same as quantizing the model weights |
| **FlashAttention 2** | A more efficient attention implementation | Does not change the precision of the model parameters |
| **Liger Kernel** | Fused Triton kernels that speed up cross entropy, RMSNorm, RoPE, SwiGLU, and more | Not a new training algorithm |
| **Checkpoint soup** | Averaging the weights of several checkpoints; the result is still one model | Not an inference ensemble that runs several models at once |
| **RFT/STaR** | Sample offline, verify, keep the correct answers, then run SFT again | Not policy-gradient RL |
| **GRPO** | Sample several completions online and update the policy with the relative reward inside the group | Not the same as rejection sampling |

A typical QLoRA run used `load_in_4bit=True`, NF4, double quantization, BF16 compute, LoRA, FlashAttention 2, gradient checkpointing, and `paged_adamw_8bit` all at once. These are separate memory layers, and they stack.

The methodology skill's cards carry the full method for RFT/STaR and GRPO; the framework skill's cards carry Liger in full and show BitsAndBytes and FlashAttention in use inside each trainer that supports them. This card keeps only the name-level difference the workflow needs.

The next layer is the evaluator, the output contract, and inference, covered by the cards that follow.
