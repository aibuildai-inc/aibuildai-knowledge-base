# 5th place solution

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #5
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/574262

First, I would like to express my sincere gratitude to XTX Markets and Kaggle for hosting this excellent competition. Organizing an event of this quality and scale is a significant undertaking. I also extend my appreciation to all the participants for their dedication and contributions.

**Solution Details (luck + 14b-awq + lmdeploy):**

*   **Public Score:** 28
*   **Private Score:** 29
*   **Model:** Deepseek-R1 Distill-Qwen-14B-AWQ ([https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-14b-awq](https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-14b-awq) - MIT License)
*   **Inference Server:** lmdeploy ([https://github.com/InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) - Apache 2.0 License)
*   **Notebook:** [https://www.kaggle.com/code/stuveee/aimo-lmdeploy](https://www.kaggle.com/code/stuveee/aimo-lmdeploy)
*   **Key References:** [https://www.kaggle.com/code/yekenot/aimo-2-deepseek-r1-distill-qwen-7b-awq](https://www.kaggle.com/code/yekenot/aimo-2-deepseek-r1-distill-qwen-7b-awq) (public notebook of Vladimir Demidov)
*   **E2E handler for HF Endpoints:** https://huggingface.co/usernameeeeeeee/r1-distill-14b-awq-casperhansen-e2e-solution-handler
* **Computational Resources (to reach the winning submission):** Kaggle's L4x4 GPU (15h/week)

## Overview

Compared to Vladimir Demidov's public notebook, I utilized the 14B-AWQ version of the DeepSeek-R1 Distill-Qwen model and implemented inference using lmdeploy, resulting in a 28% throughput gain over vllm. I learned these from the public model and discussion shared by the imagination-research team – whose insights were greatly appreciated.

I attribute my 5th place finish largely to fortunate circumstances, as a near identical submission achieved a public score of only 23. I share this outcome with those who may have more contributions or developed stronger models, yet did not experience the same favorable results. I hope everyone has the opportunity to benefit from a similar stroke of luck in future events.

Finally, I am also sharing a simulation technique applied during this competition, which reduces evaluation compute from O(N) to O(1). To avoid potential submission waste or a slight performance downgrade, I directly adopted answer extraction/voting codes and prompts from Vladimir Demidov's notebook (though I had my own version). I am very grateful for the original author(s) for their work.



## Throughput


Compared to vllm, lmdeploy increases the throughput by 28% in the 14b-awq setting.


|||lmdeploy|sglang|vllm 0.7.2|||||||||
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|14b-awq bs 9 seqlen 13500|Question Time| 400s|462s|511s|
||Throughput|304 token/s | 263 tokens/s|238 tokens/s|
|32b-awq bs 9 seqlen 9000|Question Time| 355s|374s|396s|
||Throughput|228 token/s | 193 tokens/s|183 tokens/s|
|7b-awq bs 32 seqlen 13500|Question Time| 403s|391s|505s|
||Throughput|1073 token/s | 1103 tokens/s|856 tokens/s|


* Throughput testing code: https://www.kaggle.com/code/stuveee/aimo2-throughput-sglang-lmdeploy-vllm


* I gave up on vllm engine v1 during the competition because I couldn't reproduce Vladimir Demidov's score on public lb with engine v1 and there were also accuracy downgrade problems reported in the forum. (Later, it seemed that overfitting was the actual cause.)

* note: according to team imagination-research's and tascj's great solution writeups, using int8 kv quantization can further enhance throughput without sacrificing accuracy.




## Details and Strategies


<!-- **Running details:** -->
### Running details:

* temperature: 0.9
* top_p: 0.9
* min_p: 0.05
* stop_words: ['&lt;&#x2F;think&gt;']
* batchsize: 9
* seqlen: 13500

### Strategies:

* To avoid complexity, I use the same batchsize and cutoff length for all questions, just to ensure the entire run to complete in about 4.5~4.7h.

* For each question, 9 requests run concurrently, and cutoff at seqlen 13500. Majority vote is applied to get the answer.

* Note: a question-level early stop strategy saves about 4.5~9% tokens without losing accuracy (see next section). However, it might bring bugs and implementing it in lmdeploy turbomind seemed effort consuming during the competition, so I didn't apply it.


## Simulations

### Overview:

Doing simulations can help saving compute resource from O(N) to O(1) by alternating the evaluation steps. It helped me to save a lot of submissions and compute resources during the competition.

Vanilla Evaluation:
``` markdown
For each new scheme:
    Use llm to generate sufficient samples of this scheme
    Obtain evaluation result
```

Simulation:
``` markdown
Use llm to generate sufficient samples as the 'groundtruth' 
For each new scheme:
    Sample from the 'groundtruth' the outputs of this scheme (until the evaluation result converges)
    Obtain evaluation result
```

Note that step 1 of simulation only needs to be conducted once, and there is no need to generate new outputs when evaluating new schemes. Also, more samples lead to a more accurate 'groundtruth'.



code: https://www.kaggle.com/code/stuveee/aimo2-simulation


I conducted simulations for the purposes including:
* grid search - batchsize & seqlen
* grid search - temperature
* estimate early stop benefits and losses


## Grid Search - Batchsize & Seqlen:

I empirically used max_question_time = 410s to estimate the seqlen and batchsize that ensures the entire run to finish in 4.5~4.7h, resulting in:
|Batchsize|5|6|7|8|9|10|11|12|13|14|15|16|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|seqlen|20000|17000|15500|14500|13800|13300|12800|12200|11600|11100|10600|10200|



* The simulation runs show that the optimal (batchsize, seqlen) is at (10, 13300) or (9, 13800).



### Grid Search - Temperature:

* Temperature 0.9 seems to have an observable advantage over temperature 0.8 for the 14b model:


* In the meantime, temperature 0.85/0.95 do not seem better than 0.9:





### Estimate Early Stop Benefits and Losses:


* According to the simulation, at batchsize 9, a radical early stop strategy saves about 9% tokens, while a conservative strategy saves about 4.5%. Both the radical strategy and the conservative strategy have no accuracy loss when batchsize < 12.



Details of the strategies are as follows:
* conservative
    * stop when: running_requests + 2nd_most_answer <= 1st_most_answer
* radical
    * In addition, stop when: 1st_most_answer >= current_answers * (80% - (current_answers - 5) / (batchsize - 5) * 30%)
    * i.e., linearly interpolate between:
        * 4 consistent answers when 5 are obtained
        * 50% consistent answers when all obtained


## Failed Attempts

### AWQ enhancement

**groupsize 128 -> 32:** not supported by lmdeploy

**customized AWQ:**
1. calib_data: 
    * mit-han-lab/pile-val-backup -> selective samples from lightr1-stage2.json (I appreciate LightR1 team for their work), question + solution

2. max_calib_data_seqlen: 
    * 512 -> 20000

3. increasing sample num

&nbsp;
  Note: though the original AWQ paper shows that AWQ is robust to and don't need a large scale of calibration data, I still made an attempt.

  

  The AIME25 result shows that 14b-awqv2 (i.e., customized AWQ) with 8 calibration samples performs better than 14b-awq-casperhansen. However, it couldn't outperform 14b-awq-casperhansen in my few submission attempts.



### SFT
I failed to conduct SFT because the sequence lengths for reasoning tasks were too large, and there was not enough GPU memory to fit, causing GPU OOMs even with lora.



### failure of reproducing the public notebook
I could not reproduce the score of Vladimir Demidov's notebook in my own code. I spent 20 submissions, bisected the code differences, and ended up finding that either change below leads to a score downgrade from 23~26 to 18~21:

1. merely changing vllm engine from v0 to v1
2. merely using vllm-0.6.3.post1 engine v0 without seed




## Appreciation

Finally, I would like to once again thank XTX Markets and Kaggle for hosting this wonderful competition, and to all the participants for their incredible effort and dedication. I am also deeply grateful to everyone who supported me throughout this journey. Their encouragement was truly invaluable.
