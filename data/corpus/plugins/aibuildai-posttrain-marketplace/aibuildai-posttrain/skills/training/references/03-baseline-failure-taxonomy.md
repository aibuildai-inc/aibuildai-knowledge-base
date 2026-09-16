# Baseline and failure taxonomy: the total score is not enough

Read in step 2 when breaking down the loss, and again in step 9 when re-diagnosing.

A good run does not stop at one baseline accuracy. It also records:

- the rate of valid and invalid outputs;
- the share that is correct but in an illegal format;
- the share that hits max tokens;
- the share that ends normally on EOS;
- the average output length;
- repetition loops;
- answer extraction failures;
- the tool name right but the arguments wrong;
- code syntax errors, runtime errors, timeouts, and wrong answers;
- each axis of a rubric-scored task;
- the score per question type or per source category.

## Cases

Suppose a math-contest run finds that when the model finishes inside the token budget, most of its answers are correct, and the real main loss mode is that **most outputs are truncated**. The move is not to add more long chain-of-thought; it is to switch to the shortest correct traces and make the final `ANSWER` line stronger.

A rubric-scored health-advice run might find the base model already has some medical fact knowledge while completeness and communication sit near zero. What it needs is the full answering behavior of health question answering, not more medical facts.

A strong code-generation run starts from its measured baseline and then measures SFT, STaR, and several GRPO checkpoints in turn, instead of only watching the training loss.

## Failure classes worth writing into a skill

```text
FORMAT_FAILURE
TERMINATION_FAILURE
TRUNCATION_FAILURE
REPETITION_FAILURE
KNOWLEDGE_FAILURE
REASONING_FAILURE
ARGUMENT_BINDING_FAILURE
EXECUTION_FAILURE
INSTRUCTION_FOLLOWING_FAILURE
CATASTROPHIC_FORGETTING
```

A later trick should be triggered by a failure class, not run every time.

The methodology skill's cards on STaR and GRPO carry the full method; this card keeps only the measurement the workflow needs.
