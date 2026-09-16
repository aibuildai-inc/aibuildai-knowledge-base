# Will the saved model load

These rules belong to whatever program loads a model next - an evaluator, a serving engine, another training stage - not to the library that trained it. That is why they sit in one file instead of being repeated on every library card. Read this before the first save, not after the run: every rule here can be checked in seconds, and each one costs a whole run when it is found late.

Each rule names the card it was read from. Open that card for the exact commit and the source it quotes.

## The one that ends the most runs: an adapter directory is not a model

Parameter-efficient training (LoRA and its relatives) trains a small adapter on a frozen base model, and saving it writes the adapter alone. What lands on disk is `adapter_config.json` plus a `adapter_model.safetensors` of a few megabytes, and a plain `from_pretrained()` on that directory does not give you the trained model (`huggingface__peft.md`, `huggingface__trl.md`). The base model has to be paired back in at load time, from the base model id recorded inside `adapter_config.json`.

- Reload for evaluation: `PeftModel.from_pretrained(base_model, adapter_path)` or `AutoPeftModelForCausalLM.from_pretrained(adapter_path)` (`huggingface__peft.md`).
- Merge instead, when the loader must see one ordinary model directory: `model.merge_and_unload()` then `save_pretrained(...)`. Merging is one way: after it you cannot unmerge, load several adapters, or switch the adapter off, and not every method supports merging at all (`huggingface__peft.md`).
- Every library expresses this in its own words, and the words matter: LlamaFactory merges through a separate `llamafactory-cli export` step and warns not to merge a quantized model (`hiyouga__LlamaFactory.md`); Unsloth writes a merged model with `save_pretrained_merged(...)` and steers you away from the plain PEFT reload path (`unslothai__unsloth.md`); verl's `checkpoint.save_lora_only=True` cuts a 27B checkpoint from about 54 GiB to about 150 MiB and loads back with `strict=False`, which is a partial state by design (`verl-project__verl.md`).
- A serving engine will not rescue you here. vLLM loads a standard Hugging Face model directory or repository id and nothing else, so whether a run's output can be served is entirely the training library's contract (`vllm-project__vllm.md`).

## The one that makes a resume impossible

`save_only_model` writes the weights and drops the optimizer, scheduler, and RNG state. It shrinks a checkpoint a lot, and it means the run can never be resumed from it: the docs say plainly that with this option the model can only be loaded through `from_pretrained` (`huggingface__trl.md`), and LlamaFactory's own docs say the result suits inference, evaluation, or weight conversion rather than a strict resume (`hiyouga__LlamaFactory.md`). Turn it on only when you know the run will not be continued.

Retention is a second way to lose a resume: `save_total_limit` deletes older checkpoints, and in a cluster trainer the same job is done by `max_ckpts_to_keep`, which drops whole checkpoint directories rather than trimming what is inside one (`huggingface__trl.md`, `NovaSky-AI__SkyRL.md`).

## Sharded checkpoints are not loadable models

A distributed trainer writes per-rank shards, not a Hugging Face directory, and something has to convert them.

- verl writes `global_steps_N/actor/` with per-rank `model_world_size_*_rank_*.pt` files, and turns them into a loadable model with `python -m verl.model_merger merge --backend {fsdp,megatron} --local_dir <checkpoint> --target_dir <output>`. Its FSDP layout binds `model`, `optimizer`, and `extra` together for save and load, and an older layout is rejected outright at load time with a migration script offered (`verl-project__verl.md`).
- SkyRL keeps the resumable checkpoint and the loadable export apart: the checkpoint carries `model_state.pt`, `optimizer_state.pt`, and `lr_scheduler_state.pt`, while the Hugging Face export is a separate optional artifact controlled by `hf_save_interval`, which defaults to disabled. A run that never sets it finishes with nothing an evaluator can open (`NovaSky-AI__SkyRL.md`).
- The same card records that the reference model is deliberately not checkpointed; it is rebuilt from the policy on resume, so its absence from the directory is not damage (`NovaSky-AI__SkyRL.md`).

## What the saved folder tells the grader

The model's delivery folder is the only channel to the program that scores it. A checkpoint's `huggingface/` subdirectory carries `config.json`, `tokenizer_config.json`, and `generation_config.json` (`NovaSky-AI__SkyRL.md`), and that last file decides how the model speaks at scoring time. A generation setting saved next to the weights is picked up by whatever loads the folder, so it can cap or change every later evaluation without any warning. Read it after the save and decide every field on purpose; the methodology skill's `references/inference.md` holds the measured evidence for how large that effect is.

Two more things travel in the folder and break silently. A chat template or an end-of-sequence token that is wrong at export time shows as good results in the trainer and gibberish or endless generation somewhere else (`unslothai__unsloth.md`). And in a colocated RL setup, a serving engine woken from sleep mode with one attention backend produced silently wrong generations rather than an error (`vllm-project__vllm.md`).

## The pre-flight check

**This checklist is our own composition.** No first-hand source publishes a positive "your checkpoint will load" list; engines and libraries document only what they refuse, and turning those refusals around is our work. Run it before the long training starts, so a failure costs seconds instead of hours.

1. Build the model with the exact adapter and quantization settings the real run will use, and train one step.
2. Save a checkpoint with the exact save settings of the real run - the same `save_only_model`, the same adapter-only flag, the same export interval.
3. List what landed on disk. If you see only `adapter_config.json` and `adapter_model.safetensors`, decide now whether the evaluator gets the base model too, or whether the run must merge.
4. Load that checkpoint with the program that will score the final model, not with the trainer that wrote it.
5. Generate one answer with it and read `generation_config.json` in the saved folder against what you meant to ship.
6. If the run may need to be continued, resume from that one-step checkpoint before trusting that it can be resumed later.

## What this file does not cover

No serving engine's refusal code was read first-hand for this file: the rules above come from the training libraries' own cards and their quoted documentation, not from the loader's source. Rank limits, quantization limits, and required-file checks inside a serving engine are therefore not listed here, and step 4 of the pre-flight check is what stands in for them.
