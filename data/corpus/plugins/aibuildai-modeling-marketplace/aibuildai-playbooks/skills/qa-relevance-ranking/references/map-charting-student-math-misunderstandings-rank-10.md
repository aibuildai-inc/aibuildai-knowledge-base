# 10th Place Solution

Competition: map-charting-student-math-misunderstandings
Rank: #10
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/10th-place-solution

Huge thanks to the organizers and Kaggle — my first medal-focused entry, and my first-ever medal: a solo gold 🏅. Below is a concise summary of the approach.

## Overview

Treat the task as direct multi-class classification and ensemble 15 QLoRA-finetuned models (5-fold × 3 backbones: Qwen3-Reranker-8B, Qwen3-Embedding-8B, Qwen2.5-32B-Instruct).

Key ingredients

- Data hygiene & fold control: auto-fix True/False inconsistencies; keep near-duplicates in the same fold.
- Focal + CE mixed loss with class-weight warmup (~33%) to stabilize long-tail classes.
- Inference acceleration: for Qwen2.5-32B-Instruct, merge LoRA → split base/cls-head → GPTQ-4bit base + FP16 classifier head → vLLM embed → logits → ensemble.

------

## Models & Training

### Backbones (all via `AutoModelForSequenceClassification`)

1. Qwen3-Reranker-8B (QLoRA)
2. Qwen3-Embedding-8B (QLoRA)
3. Qwen2.5-32B-Instruct (QLoRA)

Each backbone is trained with 5 folds ⇒ 15 models total.

### Loss & optimization

- Loss: a mixture of Focal (computed on unsmoothed CE to emphasize hard examples) and Cross-Entropy with label smoothing = 0.1 for calibration; per-class weights applied after warmup.
- Warmup: `weight_warmup_ratio ≈ 0.33` of total steps to avoid unstable early updates on the long tail.
- Quant-aware training: the 8B models use bitsandbytes 4-bit NF4 + gradient checkpointing (QLoRA-style).

### Hyperparameters

- Qwen3-Reranker-8B / Qwen3-Embedding-8B (single RTX 4090)
   `lr=1e-4`, `epochs=4`, scheduler=`cosine`, `weight_warmup_ratio=0.33`, effective global batch size=16.
- Qwen2.5-32B-Instruct (A800-40GB)
   `lr=1e-4`, `epochs=3`, `weight_warmup_ratio=0.33`, effective global batch size=32.

LoRA config (typical): `r=64, alpha=128, dropout=0.05`, target Q/K/V/O and MLP projections; classifier head (`score`) is saved.

------

## Inference Engineering

**8B models**

- Do not merge LoRA. Save a 4-bit quantized base and load it with the LoRA adapter for inference so a single GPU can handle each model efficiently (also friendly to dual-T4 scheduling).

**32B model**

- Workflow: merge LoRA → split base & classifier head → GPTQ-4bit quantize the base (keep cls-head in FP16) → run vLLM in `embed` mode → apply an external Linear head to produce logits → write logits.

```python
def split_model(args):
    """
    将一个完整的序列分类模型拆分为基础模型（主干）和分类头。
    """
    print(f"[*] Step: Splitting model...")
    print(f"    - Merged model path: {args.merged_model_path}")
    os.makedirs(args.base_model_save_path, exist_ok=True)
    os.makedirs(args.classifier_head_save_path, exist_ok=True)

    full_model = AutoModelForSequenceClassification.from_pretrained(
        args.merged_model_path, 
        device_map="cpu", 
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.merged_model_path)

    base_model = full_model.model
    classifier_head = full_model.score

    print(f"    - Saving base model to {args.base_model_save_path}...")
    base_model.save_pretrained(args.base_model_save_path)
    tokenizer.save_pretrained(args.base_model_save_path)

    print(f"    - Saving classifier head to {args.classifier_head_save_path}...")
    torch.save(classifier_head.state_dict(), os.path.join(args.classifier_head_save_path, "classifier_state_dict.bin"))

    classifier_config = {
        "hidden_size": full_model.config.hidden_size,
        "num_labels": full_model.config.num_labels,
        "id2label": full_model.config.id2label,
        "label2id": full_model.config.label2id,
        "bias": classifier_head.bias is not None,
    }
    with open(os.path.join(args.classifier_head_save_path, "classifier_config.json"), 'w') as f:
        json.dump(classifier_config, f)
        
    import shutil
    shutil.rmtree(args.merged_model_path)
    print(f"    - Removed merged model path: {args.merged_model_path}")
    
    print("[*] Step: Splitting model finished successfully.")
    
def run_inference(args):
    from vllm import LLM
    """
    加载量化后的模型和外部自分类头，使用vLLM进行推理，并将结果保存到JSON文件。
    """
    print(f"[*] Step: Running inference with vLLM...")
    print(f"    - Quantized model path: {args.quantized_model_path}")
    print(f"    - Classifier head path: {args.classifier_head_path}")

    with open(os.path.join(args.classifier_head_path, "classifier_config.json"), "r") as f:
        classifier_config = json.load(f)

    classifier_head = nn.Linear(
        classifier_config['hidden_size'],
        classifier_config['num_labels'],
        bias=bool(classifier_config.get("bias", True))
    )
    state = torch.load(os.path.join(args.classifier_head_path, "classifier_state_dict.bin"), map_location="cpu")
    classifier_head.load_state_dict(state)
    classifier_head.to("cuda").eval()

    llm = LLM(
        model=args.quantized_model_path,
        quantization="gptq",
        dtype="half",
        task="embed",
        override_pooler_config={"pooling_type": "LAST", "normalize": False},
        tensor_parallel_size=1
    )
    print("    - vLLM engine initialized (task=embed).")

    le = LabelEncoder()
    train_df = pd.read_csv(args.label_data_path)
    train_df.Misconception = train_df.Misconception.fillna('NA')
    train_df['target'] = train_df.Category + ":" + train_df.Misconception
    le.fit(train_df['target'])
    
    all_results = []
    try:
        embed_outputs = llm.embed(TEST_PROMPTS)
        embeddings = torch.from_numpy(np.stack([np.array(o.outputs.embedding, dtype=np.float32) for o in embed_outputs])).to("cuda")
        
        with torch.no_grad():
            logits = classifier_head(embeddings)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            top_indices = np.argsort(-probs, axis=1)

        for i in range(len(TEST_PROMPTS)):
            pred_id = top_indices[i, 0]
            top_3_ids = top_indices[i, :3]
            result = {
                "prompt_id": i,
                "prompt_text": TEST_PROMPTS[i],
                "predicted_label": le.inverse_transform([pred_id])[0],
                "predicted_score": float(probs[i, pred_id]),
                "top_3_labels": le.inverse_transform(top_3_ids).tolist(),
                "top_3_scores": probs[i, top_3_ids].tolist()
            }
            all_results.append(result)
            print(f"    - vLLM Prediction for prompt #{i}: {result['predicted_label']} (Score: {result['predicted_score']:.4f})")
    
    finally:
        del llm, classifier_head; gc.collect(); torch.cuda.empty_cache()

    with open(args.output_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"    - vLLM inference results saved to: {args.output_json_path}")
    print("[*] Step: Inference finished successfully.")
```

- Quantization: GPTQ-4bit with `group_size=64`, symmetric, GAR (Group-Aware Reordering), `true_sequential=True`; ~2k calibration texts formatted like training prompts.
- vLLM settings:
   `quantization="gptq"`, `dtype="half"`, `task="embed"`,
   `override_pooler_config={"pooling_type": "LAST", "normalize": False}`.

------

## Ensembling

I used a weighted logit blend—weight 2 for each 32B fold and 1 for each 8B fold—then took Top-3 for MAP@3, which proved more stable than probability averaging.

------

## What Didn’t Work (for me)

- Hierarchical multi-task classifier with hard logic constraints
- 32B model as pointwise reranker
- 32B model as listwise reranker
- TTA

Given the label noise, long tail, and MAP@3 objective, clean splits + balanced training + many modest models beat heavier pipelines.

------

## Results

| Setting                | Public MAP@3 | Private MAP@3 |
| ---------------------- | ------------ | ------------- |
| Single 8B model        | 0.946        | 0.942         |
| 8B backbones × 5 folds | 0.950        | 0.946         |
| 32B-Instruct × 5 folds | 0.950        | 0.947         |
| Ensemble (15 models)   | 0.951        | **0.948**     |

----

## Acknowledgments & References

Many thanks to the Kaggle community for the discussions, notebooks, and write-ups that informed my data hygiene, training, quantization, and ensembling choices. In particular:

- [brendanartley](https://www.kaggle.com/brendanartley) — [discussion](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/596778#3271298)
- [kishanvavdara](https://www.kaggle.com/kishanvavdara) — [notebook: Ensemble Gemma/Qwen/DeepSeek](https://www.kaggle.com/code/kishanvavdara/ensemble-gemma-qwen-deepseek)
- [cdeotte](https://www.kaggle.com/cdeotte) — [notebook: Gemma2-9B-IT CV 0.945](https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945)
- [jaytonde](https://www.kaggle.com/jaytonde) — [notebook: DeepSeekMath-7B LB 0.944](https://www.kaggle.com/code/jaytonde/deepseekmath-7b-lb-0-944)
- [ebinan92](https://www.kaggle.com/ebinan92) — [5th-place solution write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/ebi-ktr-5th-place-solution)
- [yannan90](https://www.kaggle.com/yannan90) — [8th-place solution write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/yannan-chen-8th-place-solution)
- [sayoulala](https://www.kaggle.com/sayoulala) — [discussion](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519)
- [nikitababich](https://www.kaggle.com/nikitababich) — [6th-place solution write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/nikita-babich-6th-place-solution)
- [conjuring92](https://www.kaggle.com/conjuring92) — [1st-place detailed solution)](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution)
