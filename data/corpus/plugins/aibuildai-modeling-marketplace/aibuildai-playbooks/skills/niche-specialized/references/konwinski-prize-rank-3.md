# 3rd Place Solution

Competition: konwinski-prize
Rank: #3
Source: https://www.kaggle.com/c/konwinski-prize/writeups/3rd-place-solution

First of all, we would like to thank the hosts for organizing such a meaningful competition.
We participated in a competition to learn how AI agents to code, and we learned a lot. As discussed in detail later, We think it was just luck that we ended up in 3rd place, but we are honored to have been able to get this position.

# Pipeline

Most parts of our solutions are based on [starter-notebook-select-patch-verify](https://www.kaggle.com/code/huikang/starter-notebook-select-patch-verify) by @huikang. Our uniqueness is that before we start solving an issue, we use the difficulty estimation of the issue to select which issues should be solved and which should be skipped.

To reduce the time required to load the LLM into GPU memory only once, we used the same model (deepseek-r1-distill-llama-70b-awq) for all runs. For difficulty estimation, we applied the LoRA adapter to the same base model.

The overall pipeline is as follows:

## 1. Difficulty Estimation

- Issue difficulty estimation using LoRA fine-tuned models
- We used the SWE-bench_Verified dataset and train a multi-class classification task of easy/medium/difficult.
- Calculate the probability that each issue is easy, and skip answering that issue if it is less than 0.5

## 2. Search Query Generation and Search

- Generate queries to find the parts needed to resolve an issue based on the issue content and the repository directory structure.
- Set the LLM temperature parameter to 0.6 and run 6 parallel runs
- Take the line that matches the query and the 12 lines before and after it, and use them as input for patch generation.

## 3. Patch Generation

- For each of the six search results, generate six patches using the issue and the search results as input.

## 4. Verification

Check if the patch is correct by the following process. If all six are incorrect, skip the answer to that question. If there is one that passes all the checks, select one of them as the answer:

- LLM judgment: Use LLM to judge whether the patch is correct with Yes/No
- Format validation: Determines whether a unidiff.PatchSet can be parsed as a patch and contains one or more changes.
- Dry-run test: Run the patch command dry to check if it can be applied without errors.

# Training for Difficulty Estimation

We used LoRA (Low-Rank Adaptation) fine-tuning for difficulty estimation.

## Datasets and Splits

- Dataset: SWE-bench verified (500 questions in total)
- Split: 400 questions for training, 100 questions for validation

## Class definition

- Easy: Can be fixed in under 15 minutes
- Medium: 15 minutes to 1 hour
- difficult: more than 1 hour

## Model Architecture

- Base model: deepseek-r1-distill-llama-70b-awq
- Method: LoRA adaptation with rank 32
- Task: 3-class classification (easy/medium/difficult)


# Conclusion

In this competition, there is a big penalty for incorrect answers, so there was a big advantage in avoiding incorrect answers. Although we avoided unsolvable problems by estimating the difficulty level, it would have been better to use other methods to avoid unsolvable problems in advance, avoid submitting inappropriate patches, etc., and improve the score.

Above all, like many participants, @huikang 's starter notebook was essential to our solution and served as the basis for many other discoveries. We would like to express our deepest gratitude.
