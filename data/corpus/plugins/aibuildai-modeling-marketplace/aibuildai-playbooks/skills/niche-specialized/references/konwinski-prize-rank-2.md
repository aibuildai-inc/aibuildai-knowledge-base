# Public 2nd Place Solution

Competition: konwinski-prize
Rank: #2
Source: https://www.kaggle.com/c/konwinski-prize/writeups/camaro-public-2nd-place-solution

Thanks for hosting such an interesting and practical competition. I was also drawn to SWE-Bench[1], though I had some doubts about its validity due to certain limitations and issues. This competition gave me a chance to address those concerns, and I spent almost 3 months working on this solution!

By the way, large part of this solution was written by LLM. Why not?

## TL;DR
- Agentless-based pipeline  
- Localize -> Generate Test -> Create Patch -> Verify  
- Several checks to decide whether to skip an issue and prevent submitting incorrect patches  


## Pipeline Flow Chart



## Details

The overall flow is based on Agentless[2], and I borrowed a lot of code from their repository.  

### 1. Pipeline

#### 1-1. Issue Evaluation

The first step is to look at the issue on its own to decide if it’s even solvable. Although this check rarely makes a big difference in the final score (since you can only tell if it’s solvable by trying), it does help by quickly skipping over issues that are unlikely to work out. In practice, this means saving time for cases that really matter.
Somehow the confidence level provided by the LLM can vary a lot. Qwen tends to be cautious, while Gemini is overly confident. How you phrase the prompt also makes a huge difference. For example, “Let me know if there's any problem” is very different from “Are you sure this is safe to deploy in production?” Because we needed to solve as many issues as possible, I opted for a more lenient setting.

#### 1-2. File Localization

It lists all Python files in the repository and picks out 5 candidates that might be relevant. I tried a few tweaks like increasing the candidate count from 5 to 10, doing a second pass to check for missed files, or even using embeddings for the search. None of these adjustments really helped. Even giving the model every ground truth file didn't boost the score much. It seems that if the model can’t spot the necessary files from the start, there is not much it can do.

#### 1-3. File Selection

For the 5 candidate files, the next step is to open each file and check its contents to decide if it’s truly needed. Instead of simply answering yes or no, extracting the parts that might need changes worked better. This not only keeps the context from getting too long, but also reduces the risk of the model making unnecessary modifications when it isn’t clear what should be fixed.

#### 1-4. Test Generation

This is arguably the most important part of the pipeline. In this module, we generated reproduction tests—tests that first reproduce the issue before applying the patch and then confirm that the issue is resolved after the patch is applied. Adding a few well-crafted test examples as few-shot examples significantly boosted performance, but adding more beyond a certain point didn't yield any additional benefits. If the test generation fails (for example, due to syntax errors), the module feeds back the error and tries to fix it. I also tracked a metric called “good_test_rate” (how often tests both reproduce the issue and are resolved by the ground truth patch) to improve test quality.

#### 1-5. Repair Generation

Using the project overview, issue description, and file contents, this step generates a patch in a function-calling style (specifying the file path, the old string, and the new string). If the patch fails to apply, it feeds back the failure and retries. When the patch works, the system runs the tests and stops if it's confirmed that the issue is resolved in reproduction tests. If not, it retries with the new feedback. For tougher issues, multiple attempts (up to five) are allowed, but more tries can sometimes lead to a higher failure rate.

#### 1-6. Repair Verification

After generating a patch, the next step is to make sure it’s really safe and effective. This step checks three things: whether the test logic is verified, if the issue is resolved, and if the patch doesn’t introduce regressions. Trying to be too strict here can cause every patch to fail (for example, the LLM might always say “No” if asked whether the test cases are sufficient). In the end, only patches that pass these checks are submitted.

#### 1-7. Time Management

A total timeout of 25 minutes is enforced (with a 5-minute buffer for unexpected delays), though most tasks only take 5–7 minutes. This is to catch any extreme cases, such as unusually long test execution times or getting stuck in an endless loop during generation.


## 2. LLM

I used Qwen2.5-Coder-32B[3]. After comparing it with Qwen2.5 72B[3], QwQ-32B[4], and Deepseek-R1[5], Qwen2.5-Coder-32B proved to be the most stable and fast. Since the pipeline can skip really tough issues, deep reasoning wasn’t as critical. However, a strong base LLM still matters. A significant difference was observed between the 14B and 32B versions. Early on, I experimented with Qwen2.5-14B[3], which turned out to be a mistake because I spent a lot of time to tune prompt to fix the issue that the weaker model lost context and caused more syntax errors, but changing the LLM to 32B almost completely fixed the issue. I hosted an OpenAI API-compatible server using vLLM[6], specifying a JSON schema for structured output to improve output stability. No public API supported structured output with a 130k token context, so I rented two RTX 3090s on vast.ai to run local evaluations.

## 3. Development

### 3-1. Baseline
I started by exploring several high-scoring methods on SWE-Bench, and Agentless stood out as the simplest and most developer-friendly approach. While the overall code became somewhat complex, I managed to extract the key modules and assemble them to work with SWE-Bench. 

### 3-2. Prompt Engineering
I adopted a LLM consortium style prompt engineering - by comparing outputs from multiple LLMs (including ChatGPT, Grok, Gemini, and DeepSeek) to pick the best ideas. I also tried Anthropic’s prompt generation, but its Claude-specific style (like overusing XML) wasn’t as effective since I was working with JSON structured output. In total, I developed 4 eval_issue prompts, 3 localize_file prompts, 7 select_file prompts, 14 generate_test prompts, 12 generate_patch prompts, and 4 verify_patch prompts, and the adopted the best prompts for each task. I'll post the prompts later.

## 4. Evaluation

For evaluation, I used two datasets: a subset of 100 instances from SWE-Bench-Verified[7] and the full SWE-Bench-Verified (500 instances) dataset. On the leaderboard, my model achieved an average score of around 0.052022 over 10 late submissions (with scores ranging from 0.028080 to 0.070332). For the SWE-Bench-Verified dataset, scores were between 0.04 and 0.06. However, these numbers can be gamed; for instance, nearly half the projects in SWE-Bench-Verified are Django, so tailoring solutions specifically for Django could artificially inflate the score.

Beyond just the final score, I tracked a variety of metrics to get a clear picture of how the system was performing at every stage. Here’s a list of the metrics and what they represent:

- **score**: The overall evaluation score of the solution.
- **num_solved**: The number of instances considered solved, including those that were skipped.
- **num_tried**: The total number of samples that were submitted.
- **num_successed**: Out of the submitted samples, this counts how many achieved both f2p = 1.0 and p2p = 1.0.
- **num_failed**: The number of submitted samples that failed.
- **num_degraded**: Among the failed samples, this indicates how many had an f2p score lower than 1.0.
- **num_not_solved**: Among the failed samples, this shows how many had a p2p score lower than 1.0.
- **file_recall**: The rate at which the correct file was found in the ground truth patch during the localization step.
- **selected_file_recall**: The rate at which the correct file was selected after the file selection phase.
- **selected_line_recall**: The rate at which the correct lines were identified in the selected files.
- **reproduced_rate**: The percentage of instances in which the issue was successfully reproduced.
- **good_test_rate**: The proportion of tests that both reproduced the issue and were resolved by the ground truth patch.
- **good_test_rate_in_reproduced**: Similar to good_test_rate, but calculated only on the instances where the issue was successfully reproduced.
- **replacement_success_rate**: The success rate of matching the old string in the generated patch.
- **syntax_success_rate**: The rate at which patches, after being applied, did not introduce any syntax errors.


These metrics were helpful in guiding which improvements to make. However, enhancing each metric did not always result in a better final score. For example, while increasing the number of candidate files improved the file recall, it didn’t necessarily boost the overall score — in some cases, it even introduced unintended changes that hurt performance. Similarly, although a higher good_test_rate was beneficial, forcing too many tests sometimes led to more failures rather than better outcomes.

## 5. What didn't work

- Running all unit tests before and after applying patches didn't help for me. It turned out to be too slow and risky, so I didn't use it. 
- I also tried generating extra tests after patch generation to further filter out bad patches, but that approach was hard to control and ended up doing more harm than good. In the end, I simplified the verification step by trusting the LLM’s judgment rather than relying on an overly strict testing process.

## 6. Code

I'm planning to make the code public after the competition ends.

## 6. References

1. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* arXiv preprint arXiv:2310.06770.  
2. Xia, C. S., Deng, Y., Dunn, S., & Zhang, L. (2024). *Agentless: Demystifying LLM-based Software Engineering Agents*. arXiv preprint arXiv:2407.01489.  
3. Liu, Y., et al. (2024). *Qwen2.5: Efficient and Robust Code Generation with Improved Contextual Understanding*. arXiv preprint arXiv:2412.15115.  
4. QwenLM Team. (2024). *QWQ-32B: A New Frontier in Code Generation*. Retrieved from https://qwenlm.github.io/blog/qwq-32b/  
5. Zhang, X., et al. (2025). *R1: A Robust and Efficient Approach for [Task Description]*. arXiv preprint arXiv:2501.12948.
6. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. arXiv preprint arXiv:2309.06180.
7. OpenAI. (2024). *Introducing SWE-bench Verified*. Retrieved from https://openai.com/index/introducing-swe-bench-verified/
