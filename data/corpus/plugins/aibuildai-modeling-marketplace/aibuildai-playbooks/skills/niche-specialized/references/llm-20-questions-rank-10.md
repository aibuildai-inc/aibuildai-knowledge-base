# 10th Place Solution | A 90% coverage keyword list is ALL you need

Competition: llm-20-questions
Rank: #10
Source: https://www.kaggle.com/c/llm-20-questions/discussion/531569

# 10th Place Solution
First of all, I would like to express sincere gratitude for Kaggle for hosting this interesting competition on behalf of our team (my great teammates @raki21 @benbla)

## Solution Overview

We joined the competition relatively late, just 8 days before its conclusion, after finishing LMSYS Chatbot Arena Human Preference Predictions. We started by analyzing the game and existing approaches and found that alphanumerical bisection is the optimal strategy to maximize information gain from each question given correct response. Thus, we spent most efforts in building a keyword list that can cover the most private keywords and we experimented different prompt engineering & function calling strategies for robust collaboration with the teammate as Answerer and Questioner.

Key statistics about our keyword list coverage for private keyword set (1800 from this [dataset](https://www.kaggle.com/datasets/lohmaa/llm20-evaluation-games-collected)):
```
1M keyword list coverage: 86.56% exact match

2M keyword list coverage: 87.28% exact match, 88.65% if we consider the matching algorithm of this competition
```
It means that our agents will be able to guess the keyword around 90% of the time if the answerer is giving the correct answer for each question.

## Solution Details

### 1. **Questioner**:
At first we adopted the great notebook [agent alpha approach](https://www.kaggle.com/code/lohmaa/llm20-agent-alpha) because we saw an increasing trend of unifying "protocols". During local tests, we found that LLMs are not good at answering alphanumerical bisection questions with simple prompt that agent alpha used. So we started to use a variety of prompt engineering and found that the following template works the best:

```
"""
keyword Please follow these 2 steps to compare alphabetical order:
Step 1: take the known keyword and the comparison word and write them out with a space between each character, in lowercase.
Step 2: Now iterate through the characters of both words, until you find a mismatch. The first mismatch determines alphabetical order.
If one word ends while the other has more letters, the shorter word comes first alphabetically. 
Finally answer with yes or no.
For your reference, the alphabetical order of characters is: a < b < c < d < e < f < g < h < i < j < k < l < m < n < o < p < q < r < s < t < u < v < w < x < y < z.

Does the keyword (in lowercase) precede {keyword} in alphabetical order?
"""
```
There are several components in the prompt:

- Despite its complexity, it matches all agent alpha answerer regex
- We use COT style to decompose the task
- We give the alphabetical order at last to reduce hallucination

As a result, this prompt empowers our Answerer (powered by **microsoft/Phi-3-medium-4k-instruct**) to answer the comparison question with around 80-90% accuracy in local test. Note that this prompt still does not make other LLMs significantly better in answering this type of questions. (including the most commonly adopted models **meta-llama/Meta-Llama-3.1-8B-Instruct** and **meta-llama/Meta-Llama-3-8B-Instruct**). This also partialy explains why we only have 2 wins after alpha handshake is rejected. (this statistics is from this great [notebook](https://www.kaggle.com/competitions/llm-20-questions/discussion/529683) the metric NWhr).

To further improve our guessing efficiency, we attach a tier (from 1 to 10) for each keyword using GPT40-mini. At each round of guessing, we always guess the keyword in the remaining possible keyword set that has the highest tier.

One of our agent skipped the handshake round and directly goes with agent alpha question and use a 2M size keyword list. The other one uses 1M keyword list and keep the agent alpha handshake. Their performance exchange places several times during the competition. **Final score of first one is 1140.9 (10th), the other one is 1108.3 (12th - 13th equivalent).**

### 2. Keyword List Generation and Ranking

If we skip the handshake, the max number of keywords we can search within 20 rounds is 2^20 -> 1.048M. Consider the excluding effect for each round of guess, we can cover 2^(20+1) - 2 = 2.046M. 

**EDIT:** For an empirical experiment and formal proof of this number, you can refer to the discussion thread [here](https://www.kaggle.com/competitions/llm-20-questions/discussion/531569#2976594). In short the maximum size of keyword list that bisection can guaranteed to cover is 2^(i+1) - 2 for i games.

We formulate the keyword generation task as a **supervised learning** task where we conduct a (0.8, 0.2) train-test split for all publicly known keyword. We did this in an iterative manner and combine them with the unigrams/bigrams from Wikipedia dump. We also conducted several filtering. Finally, we use GPT40-mini to assign tier for GPT-generated keywords and use rescaled frequency of n-grams for those keywords from Wikipedia. The details are summarized in the figure below:



Below are some visualizations of the private keywords that are covered in our 2M list (around 90% of total 1800 private keywords).







There is a debate in our team on whether we should keep the public keywords or not. We end up with a more prudent approach: we include the keywords in keywords.py as tier 1 (lowest tier) and scraped keywords in public set as tier 4. This enables us to remain in the top 15 from the very beginning of the competition till the end.

### 3. Answerer

Like mentioned above, the answerer is powered by **microsoft/Phi-3-medium-4k-instruct**. It takes several hacks to make it run properly in Kaggle environment single GPU. If anyone is interested, we will write a public notebook detailing the steps (our current code base is a little complex).

The answerer also adopts COT in system prompt:
```
"You are playing 20 Questions games. You are Agent Alpha answerer. Please use common sense thinking. When the keyword might relate to multiple things, use the most common one. All keywords are included in **things** category and very rarely fall into **places** or **people** category.\n\nWhen the answer is not clear or depends on the situation, please answer no. For the type of question that has the form \"Is it a A or B?\", answer yes if Is it a A is true, answer no otherwise. Please think step by step and provide yes or no at the end. End your answer when you have the conclusion.\n\n"
```

Key elements:

- Context specification to decrease hallucination
- Restrictive prompt: phi3 medium is too "creative" in answering simple questions
- COT & try to make answer brief
- We set max_new_tokens=200 to let model conduct COT in output and catch the final yes/no answer
- We implement a strict time control: when obs.remainingOverageTime < 60, just output no

Aside from prompt engineering, we write **a lot of regex expressions** to catch most types of common questions such as "does the keyword start with xxx". We use function calling to solve these prompts.

This answerer answers most non-alpha questions perfectly in local tests. However, the chance of our answerer facing a capable non-alpha agents is very low after the surging of alpha agents so we don't know its full potential.

## Reflections on this game

It is a pity that this competition ends with a non-converging behavior where a lot of good solutions does not get recognized by medals/prizes. We felt lucky and thankful for this gold medal and hope that there will be a followup competition sometime in the future where more interesting solutions can come up. As always thank you for reading our solution and hope you find it interesting somehow. I would like to express my sincerely gratitude for my fantastic teammates @raki21 @benbla again for making this competition an interesting and rewarding journey!
