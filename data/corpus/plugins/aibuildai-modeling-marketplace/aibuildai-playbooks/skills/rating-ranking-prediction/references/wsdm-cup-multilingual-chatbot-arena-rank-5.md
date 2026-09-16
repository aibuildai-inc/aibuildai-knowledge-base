# 5th Place Detailed Solution

Competition: wsdm-cup-multilingual-chatbot-arena
Rank: #5
Source: https://www.kaggle.com/c/wsdm-cup-multilingual-chatbot-arena/discussion/567856

Congratulations to all the winners! Thanks to the teammate @mianwang1024 @zhengnie233. I’d like to thank the other two teammates for teaching me what it means to sit back and reap the rewards, and revealing the greed inherent in human nature.Thanks to the organizers for hosting such an interesting competition. I'm grateful to the Kaggle community for innovative ideas and engaging discussions.I have learned a lot. Here are the detailed solution to supplement it.

code:  [5th code](https://github.com/lcy80366872/kaggle-project/tree/main/wsdm)
infer : [notebook](https://www.kaggle.com/code/linchenyu/5th-place-solution)
# **Task**
Predict which responses users will prefer  between chatbots powered by large language models (LLMs). 
A simple idea is to use a large model for binary classification. It has higher representation capability compared to smaller models and is better suited for this task. It came with a few challenges:

- Large models are very sensitive to the positions of answers A/B within the text, exhibiting positional bias.A simple idea is to use a large model for binary classification. It has higher representation capability compared to smaller models and is better suited for this task.
- Large models need to learn additional knowledge based on real data, as user preferences are not limited to the rationality of the answers, but also include the accuracy, degree of style, language, length, and fluency of the answers.
- The competition requires inferring through a large amount of data in a short period of time.

# **Thoughts**

- LoRA could be better and faster than QLoRA. A high rank can help the model emphasize more knowledge about preferences.Full training might be better than LoRA.
- Swapping the positions of answers A/B in the prompt for data augmentation would be a straightforward choice, and perhaps making additional diversity changes after the swap could be even better.
- In this scenario, soft labels may be easier to learn than hard labels, and the logits distribution may be easier to learn than the labels themselves.
- The input prompt should be specially processed to retain more information.
- Use AutoModelForCasualLM might better than AutoModelForSequenceClassification for faster inference with vllm and better reasoning.
- Post-processing might be able to address the issue where large models struggle to distinguish between two answer.

# **Data**
We aggregated multiple data sources:
[WSDM48k](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/data)
[LMSYS_55K](https://www.kaggle.com/competitions/lmsys-chatbot-arena/data)
[Add_33k](https://www.kaggle.com/datasets/abdullahmeda/lmsys-additional-33k-labelled-conversations)
[open-model(8.5k+13k+5k)](https://www.kaggle.com/datasets/nbroad/wsdm-open-models-nbroad) thanks to @nbroad 

I filtered out the duplicate data and performed data augmentation through swapping.
The final data contains 2*120k rows.

# **Model**
We try Qwen2.5-14b-it,Gemma2b-it,deepseek-r1-distill-14b. Compare with Qwen2.5-14b-it, we found that  deepseek-r1-distill-14b improve cv around 0.001 but LB drop about 0.003.(I think that's mainly due to we only generate 1 token ,deepseek-r1 will lose its reasoning ability) Gemma2b-it is lower both on it. So finally we choice Qwen2.5-14b-it.

#**Train**
## 1.Prompt
To mitigate the information loss caused by truncation, we will truncate the middle parts of the question and both responses and replace them with ellipses, similar to how "Have a nice day" would be truncated to "Have ... day".
```python
'''You are a skilled judge evaluating responses from two large language models(LLMs). Your task is to select the response that best meets the user's needs based on the query provided.

**Input Format:**
<Query>
[User's original query to both LLMs]
</Query>

<Response_A>
[First LLM's response]
</Response_A>

<Response_B>
[Second LLM's response]
</Response_B>

**Your Task:**
Carefully analyze both <Response_A> and <Response_B> in relation to the Query. Determine which response is more likely to be selected by a user based on the following criteria:
- Completeness in addressing the query
- Accuracy of information
- Clarity and coherence
- Conciseness vs appropriate detail
- Helpful examples or explanations when needed
- Professional yet engaging tone
- Sound reasoning and logic
- Format and presentation

Here is your input to process now-
Input:

<Query>
{row['prompt']}
</Query>
{'---'*10}
<Response_A>
{row['response_a']}
</Response_A>
{'---'*10}
<Response_B>
{row['response_b']}
</Response_B>

Which response is more likely to be selected by a user? (A or B)\nOutput:\n'''
```
## 2.Two stage train
[data] (https://www.kaggle.com/datasets/linchenyu/wsdm-5th-2stage-data)
###Stage 1
**Dataset**
 [WSDM48k](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/data)
[LMSYS_55K](https://www.kaggle.com/competitions/lmsys-chatbot-arena/data)
[Add_33k](https://www.kaggle.com/datasets/abdullahmeda/lmsys-additional-33k-labelled-conversations)with hard label [0,1].
**Training**
- We trained the Qwen2.5-14b-it model with LoRA in fp16 precision, directly optimizing its output probability distribution for answers A and B based on hard labels, while ignoring the loss for other tokens. 
- The LoRA parameters were set to r=64, alpha=128 (a larger r would likely be better, but we didn't have time to implement it). The learning rates were set to 1e-5 for lora_a and 5e-5 for lora_b(We referred to this [approach](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688),thanks to @conjuring92)  with a batch size of 16 and max_length set to 4096. Epochs =1.
- The training was completed in approximately 12 hours on 6 A100 GPUs.

**Predict logits**

- Performed data augmentation by swapping all the data and shuffled them in pairs as the final training data.
- Using the trained model to perform logits inference on the open-model dataset to obtain the probabilities of answers A and B.

###Stage 2
**Dataset**
 [WSDM48k](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/data)
[LMSYS_55K](https://www.kaggle.com/competitions/lmsys-chatbot-arena/data)
[Add_33k](https://www.kaggle.com/datasets/abdullahmeda/lmsys-additional-33k-labelled-conversations)with soft label like [0,1]->[0.05,0.95]
[open-model(8.5k+13k+5k)](https://www.kaggle.com/datasets/nbroad/wsdm-open-models-nbroad) with soft label infer by stage 1.
**Training**
- The same as stage 1, just need some adjustment to the loss function
- The training was completed in approximately 16 hours on 6 A100 GPUs.

# **Inferring**
We utilized VLLM for inference, loading the model in half-precision with a combined prompt and output max length of 8192, and implemented CPU offloading due to insufficient T4 GPU memory. During inference, the model was configured to generate only a single token, from which we extracted the logits for answers A and B. The answer with the higher logit value was selected as the final response. In cases where the logits for both answers were too close, we applied post-processing to determine the final response.

#**Some findings**
Here are the some of experiments on LB(Some of experiments only test on CV so they are not in the table):
| model | data| operation |LB|
| --- | --- | --- |
|qwen2.5-14b-it |wsdm|  zeroshot |0.6254|
| qwen2.5-14b-it |wsdm|  mid_cut+0.7*dpo+0.3*bce | 0.689  |
| qwen2.5-14b-it | wsdm| left_cut+lora| 0.6907 |
| qwen2.5-14b-it |wsdm|  mid_cut+random_fewshot | 0.6910  |
| qwen2.5-14b-it |wsdm|  mid_cut | 0.6938  |
| qwen2.5-14b-it |wsdm+lmsys+add_33k |  mid_cut | 0.6964 |
| qwen2.5-14b-it |wsdm|  mid_cut+swap | 0.6967  |
| qwen2.5-14b-it | wsdm| mid_cut+cot_multi_task | 0.6973 |
| qwen2.5-14b-it |wsdm+lmsys+add_33k |  mid_cut+swap | 0.7038 |
| qwen2.5-14b-it |wsdm+lmsys+add_33k |  mid_cut+swap+postprocess | 0.7039 |
| qwen2.5-14b-it |wsdm+lmsys+add_33k+open_model | mid_cut+swap +softlabel | 0.706812(final LB=0.712) |

- Here are some finding of some experiments:
**The following results were tested only on CV and did not show significant improvement:** 
- In a single input, the vector representations of the question, answer A, and answer B were extracted using the mean value, and additional contrastive learning was introduced alongside cross-entropy training during training. 
- The influence of prompt wording on training outcomes is not particularly significant.


**The following results were tested on CV and LB:** 

- Introducing random few-shot learning during training led to a slight improvement in CV but a drop in the leaderboard (LB) score. 
- A larger model was used to generate CoT, and multi-task learning was applied to the 14B model based on this CoT. The results better, with a potential 0.003 improvement. Due to an incomplete understanding of the actual mechanism, this approach was ultimately not adopted.

**The following results were tested only LB:** 

- The training max length was increased to 8192. LB dropped.
- The model was trained for two rounds.  LB dropped.
- DPO training was also added to optimize the model's preference probabilities along with BCE loss for A and B. LB dropped.
