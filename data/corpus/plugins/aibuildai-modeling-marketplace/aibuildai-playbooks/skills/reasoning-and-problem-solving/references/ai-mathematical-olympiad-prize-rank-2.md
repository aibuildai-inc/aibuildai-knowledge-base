# 2nd place solution (All code and datasets released) - CMU_MATH

Competition: ai-mathematical-olympiad-prize
Rank: #2
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/518964

# 2nd place solution: SFT + ORM

## All the code and dataset, please check our [**Github repository**](https://github.com/AIMO-CMU-MATH/CMU_MATH-AIMO).

[The notebook of the solution can be found here](https://www.kaggle.com/code/edwardsun0909/aimo-2nd-place-solution)

[policy model](https://huggingface.co/AIMO-CMU-MATH/policy-model)

[reward model](https://huggingface.co/AIMO-CMU-MATH/reward-model)

## Model Details

We fine-tuned two DeepSeek-Math-7B-RL models, one is used as policy model (to generate solutions), one as reward model (to score the solutions for weighted majority voting). The training infra is adopted from (Sun et al., 2024), bibtex is in the end.

### Policy Model

Our dataset is a combination of AMC, AIME, and Odyssey-Math. We only select those problems of integer answers since the answer of the test problems will be integers. We also delete the choices of the multiple choice problem in AMC.

- [AMC, AIME datasets can be found here](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/492945). Thanks  [@alexryzhkov](https://www.kaggle.com/alexryzhkov) !

- [Odyssey-Math dataset can be found here](https://github.com/protagolabs/odyssey-math)

We prompt GPT4 to sample code to solutions to the problems in our dataset using few-shot examples, and select the correct solutions as our training dataset for our policy model.

The datasets can be found at our github [AIMO-CMU-MATH](https://github.com/AIMO-CMU-MATH/AIMO-2nd-Solution).

We finetune the model for 3 epochs with learning rate 2e-5.

### Reward Model

#### Problem Set

We use the problems in [MATH](https://github.com/hendrycks/math/), AIME, AMC and Odyssey-Math with nonnegative integer answers as our problem set for training the reward model.

#### Reward Dataset Collection

##### Observation

On MATH dataset, if we sample solutions with DeepSeek-MATH-7B RL, we will only get correct solutions for some problems, and those solutions can be similar. On the other hand, sampling solutions with DeepSeek-MATH-7B Base fails to output any correct solutions on some problems.

On AMC, AIME and Odyssey, the observation is similar. When using DeepSeek-Math-7B-RL to sample solutions, most of the results are wrong; When using our fine-tuned policy model, the accuracy is high. 

We believe that in order to train a good reward model, the dataset should satisfies:

- Diversity: The solutions for each question are quite different.

- Label balance: The dataset contains both correct solutions (positive examples) and incorrect ones (negative examples) for each question.

Therefore, we adopt the following sample strategies.

##### 1. Interpolation

For MATH dataset, we interpolate the model parameters of DeepSeek-Math-7B RL and DeepSeek-Math-7B Base to get models with different level of capabilities. Denote the model parameters of DeepSeek-Math-7B RL and DeepSeek-Math-7B Base as $M_{RL}$ and $M_{Base}$, respectively. The interpolated model parameter $M_{\alpha}$ is a linear combination of $M_{RL}$ and $M_{Base}$.
$$
M_{\alpha}=\alpha M_{RL} + (1-\alpha)M_{Base}
$$
We choose 8 different models with $\alpha=0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0$. Each model generates two solutions for each question. The larger $\alpha$ is, the more correct answers the model samples.

By interpolation, we can get different models which yield diverse solutions, and the proportion of positive labels and negative labels are much more balanced compare to sampling with only one model.

##### 2. Fine-tuning

For AMC, AIME, Odyssey, we use the training dataset we collect for the policy model, and train the DeepSeek-Math-RL 7B with 0, 1, 2, and 3 epoch(s) to get 4 different models. Each model generates 12 solutions for each problem.

In this way, we obtain different solutions and each problems are more likely to have both correct solutions and wrong solutions.

##### Filtering

Once we collected all the solutions with labels, we do the filtering below:

- We remove those wrong solutions with non-integer answer, since only solutions with integer answer will be scored as a candidate answer in our weighted majority voting method.
- We keep the number of positive labels v.s. negative labels as 1:1 for each problem. 

#### What else we tried for fine-tuning reward model and policy model

- De-duplicate similar solutions in policy model training dataset.
- Include those gpt's wrong solutions in the policy model training dataset.
- Keep all the sampled solutions with labels in the reward training dataset.
- Not keep the proportion of positive and negative labels for each question in the reward dataset as 1:1, instead  a range of  1:2 to 2:1.

We find that the above methods don't result in models that perform better on our validation dataset. Therefore we use the methods described before.



## Solution

### vLLM

For the policy model, we use [vLLM](https://github.com/vllm-project/vllm) to speed up sampling and  huggingface transformer for reward model. Scoring a solution is just a forward pass and doesn't take much time. Each model uses one GPU (T4).

### Generation

We sample 42 solutions for each problem, the policy model is asked to write code. Prompt is simply problem + tool_instruction.

`tool_instruction = '\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.'`

---

If the result of code execution is not an integer, we give feedback and ask policy model to do it again, the code is below:

```
def vllm_round_inference(problem, num_sequences, T, rounds):
    prompt = problem + tool_instruction
    texts = [prompt for _ in range(num_sequences)]
    results = []
    result_texts = []
    for i in range(rounds):
        sampling_params = SamplingParams(temperature=T, top_p=1.0, n=1, max_tokens=1024, stop="```output")
        generated_texts = llm.generate(texts, sampling_params)
        next_round_texts = []
        completed = []
        print(len(generated_texts))
        for j, (text, generated_text) in enumerate(zip(texts, generated_texts)):
            new_text = generated_text.outputs[0].text
            completed = False
            if '```python' in new_text:
                try:
                    code_text = new_text.split('```python')[-1].split("```")[0]
                    code_result, CODE_STATUS = process_code(code_text, return_shell_output=True)
                    try:
                        result = int(float(code_result.strip()))
                        if result >= 0:
                            results.append(result % 1000)
                            result_texts.append((prompt + "```python" + new_text.split('```python')[-1]).split("```output")[0])
                        completed = True
                    except Exception as e:
                       code_result += "The code output is not an integer, final answer should be an integer."
                except Exception as e:
                    code_result = "Code execution fail"
                    pass
                if completed == False:
                    next_round_texts.append(text + new_text + "```output\n" + code_result + "\n```")

        texts = next_round_texts
        print("remaining text:", len(texts))
        if texts == []:
            break
    return results, result_texts
```



Where num_sequences was set to be 42, T=0.9, rounds=2.

### Weighted Majority voting

The outcome reward model (ORM) assigns a score to each solution. The input of the ORM is problem + code instead of problem + solution. We let the ORM to score the code generated by the policy model since the final candidate answers are simply execution results of the code.

```
def orm_score(reward_model, tokenizer, text):
    input_id = torch.tensor([tokenizer.encode(text)]).to(reward_model.device)
    with torch.no_grad():
        logits = reward_model(input_id).logits
        score = logits[0].mean(dim=-1).sigmoid().cpu().tolist()[-1]
    return score
```



---

To calculate the total weight of one answer, we calculate it as the geometric mean times $N$, i.e. if $N$ solutions lead to the same answer $a$, the $i$-th solution has a score of $S_i$, the weight of answer a is:
$$
N(\prod_{i=1}^n S_i)^{\frac{1}{N}}.
$$
We choose geometric mean times $N$ instead of the sum the weights because it prevents over-optimization. For example, in our experiment, the answer of the question "There exists a unique increasing geometric sequence of five 2-digit positive integers. What is their sum?" is 211. ORM gives score 0.75 to one solution with 211, but 0.85, 0.03, 0.15 to answer 50. The 0.85 can be a false positive of reward model and geometric mean prevents us from choosing such answer.

We also note that there is a tendency for the policy model to generate solutions with wrong answer 0, so we add some penalty for answer 0. Below is the code of weighted majority voting.

```
def weighted_geometric_mean_times_num(answers, scores):
    import math
    max_weight = -1
    number_weight_dict = {}
    max_weight_answer = None
    for answer, score in zip(answers, scores):
        if answer == 0:
            score /= 10
        if answer not in number_weight_dict:
            number_weight_dict[answer] = [score]
        else:
            number_weight_dict[answer].append(score)
    for answer in number_weight_dict.keys():
        score_list = number_weight_dict[answer]
        weight = len(score_list) * math.prod(score_list) ** (1 / len(score_list))
        if weight > max_weight:
            max_weight = weight
            max_weight_answer = answer
    return max_weight_answer
```



### Summary

We attribute the high performance of our notebook to the two points below:

- Stronger fine-tuned policy model
- A powerful reward model that selects the true answers from multiple candidate answers.

In the AIMO train set of 10 problems, with the fine-tuned policy model, the accuracy improves from 1/10 to 2/10, and the reward model further improves it from 2/10 to 4/10.
### Citation
The fine-tuning [code base](https://github.com/Edward-Sun/gpt-accelera) is adopted from the code of the paper below:
```
@article{sun2024easy,
  title={Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision},
  author={Sun, Zhiqing and Yu, Longhui and Shen, Yikang and Liu, Weiyang and Yang, Yiming and Welleck, Sean and Gan, Chuang},
  journal={arXiv preprint arXiv:2403.09472},
  year={2024}
}
```
