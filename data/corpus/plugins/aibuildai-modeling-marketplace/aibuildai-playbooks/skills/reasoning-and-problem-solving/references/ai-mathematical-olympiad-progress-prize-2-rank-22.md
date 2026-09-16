# 21st Place Solution

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #22
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/571289

Thanks to all participants, especially those who created awesome notebooks, models and engaging discussions. I learned a lot. Thanks to Kaggle and the Host for their continued support throughout this long, challenging but exciting competition.

I developed many approaches some of which drew inspiration from discussions and public notebooks. Below I mainly share my successful approach so far with **public score 28**.

## Final Approach: Logits Distillation from Qwen-QWQ 32B into DeepSeekR1-Distill-Qwen14B

Since the idea is to learn how  Qwen-QWQ 32B solves problems, I needed problems that are relevant for the AIMO2 competition. Hence, I included the reference set as well, see below

### Datasets
- Random 6% (random seed 42) of Problems from [OpenR1 Dataset] (https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) where DeepSeek-R1 found the right answer + AIMO2 Reference Set + [AIME2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) + [AIME2025](https://huggingface.co/datasets/opencompass/AIME2025).
- Total Dataset Size: 5274

### Main Methodology
**1. Initialize three models on shared GPUs as follows**

- Qwen-QWQ 32B-AWQ loaded via vLLM (teacher model for text generation)
- Qwen-QWQ 32B-AWQ loaded with AutoModelForCausalLM (teacher for logits computation)
- DeepSeekR1-Distill-Qwen14B (student model)

**2. Generate text, form inputs, forward and compute loss**

- For each input problem, create a prompt, including system message and ask Teacher Model to solve the problem (using vLLM for faster inference)
- Use Teacher Model (AutoModelForCausalLM instance) to get logits of tokens present in the generated text + prompt
- Call Student Model to compute logits for the same input text (again, the prompt and solution are included as inputs for logits computation)
- Compute KL divergence between student logits and teacher logits

### Training Details
```python
SYSTEM_PROMPT = """You are the most powerful math expert. Please solve the problems with deep reasoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{} after taking modulo 1000."""

MAX_SEQ_LEN = 4096 # maximum sequence length in teacher text generation

sampling_params = SamplingParams(max_tokens=MAX_SEQ_LEN, temperature=0.7, skip_special_tokens=False)

generated_text = teacher_model.generate(
            prompts=teacher_prompt,
            sampling_params=sampling_params
        )

# Now use tokenizers from teacher model and student model to get input_ids and attention_mask

teacher_logits = teacher_logits_model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask).logits

student_logits = model(input_ids=student_input_ids, attention_mask=student_attention_mask).logits

student_log_logits = torch.nn.functional.log_softmax(student_logits, dim=-1).to('cuda')
teacher_log_logits = torch.nn.functional.log_softmax(teacher_logits, dim=-1).to('cuda')

loss = torch.nn.functional.kl_div(student_log_logits, teacher_log_logits, log_target=True, reduction='batchmean')

training_args = TrainingArguments(
    output_dir="./qwen14b_distilled",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=1,
    learning_rate=1e-6,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=20,
    remove_unused_columns=False,
    report_to="none",
)
```
### Linear Weight Ensembling: **This is what really stood out**
- I kept checkpoints at different training steps and averaged model weights from those
- Averaged checkpoints are: 0.1\*ckpt-20 + 0.1\*ckpt-180 + 0.1\*ckpt-260 + 0.1\*ckpt-280 + 0.1\*ckpt-360 + 0.1\*ckpt-640 + 0.1\*ckpt-1840 + 0.1\*ckpt-2300 + 0.2\*last-ckpt

### Evaluation with Quantized Distilled Qwen14B Model

- AIME2024/2025: 43/60

- Reference Set: 7/10

- Public LB: 28 (with max_seq_len=8192*3//2 + 1024, top_p=0.9, min_p=0.05, temperature=1.0). I also used vLLM Engine class (not LLM) and could cancel requests when a given answer reaches 33% (frequency) w.r.t. ```max_num_seqs=12```

## Final Thoughts
- I could have trained longer. Unfortunately, I got the idea during the last week of the competition
- Training longer could improve performance
- Larger learning rate (1e-5) resulted in a poor-performing 7B model. Did not try it with the 14B model

## What Did Not Work:
- Supervised Finetuning with DeepSeek-R1-Distill-Qwen7B
- GRPO with DeepSeek-R1-Distill-Qwen7B
- Mergekit (slerp, sce) with my 14B model checkpoints, score dropped to 20
