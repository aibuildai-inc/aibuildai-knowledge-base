# Chat template and tokenizer alignment

Read in steps 1, 5 and 6: the token sequence the model actually sees.

This is not the same as plain "output format". It is about **the token sequence the model actually sees**.

Check these:

- whether `apply_chat_template()` matches the evaluator exactly;
- `add_generation_prompt=True/False`;
- the system, user, and assistant boundaries;
- BOS, EOS, `<|im_start|>`, `<|im_end|>`;
- the pad token and the padding side;
- whether the tool schema is inserted by the template or written into the user text;
- the default `<think>` behavior of a reasoning model;
- whether the tokenizer has an architecture-specific regex problem;
- whether the template used in training is the same as the template saved into the final model folder.

## Common mistakes

1. The training data uses the official instruct template, but the evaluator uses another, simpler template for the base model.
2. The assistant header is repeated when the prompt and the completion are joined.
3. `prompt_len` is computed in a way that does not match the final rendered text, so part of the answer is masked out.
4. The tokenizer adds a special token by itself, and the code adds it again by hand.
5. Qwen's `<|im_end|>` is not set as EOS.
6. The Gemma, SmolLM, and Qwen templates get mixed together by mistake.

A QLoRA run can first render the prompt alone and then the prompt plus the answer, and then set the first `prompt_len` labels to `-100`.

The framework skill's cards carry the full tokenizer and trainer detail per library; this card keeps only the alignment check the workflow needs.
