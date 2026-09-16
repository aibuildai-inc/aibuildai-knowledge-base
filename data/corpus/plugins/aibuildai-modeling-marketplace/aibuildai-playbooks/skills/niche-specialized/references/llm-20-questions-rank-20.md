# [20th] A simple solution, attempts and thinking.

Competition: llm-20-questions
Rank: #20
Source: https://www.kaggle.com/c/llm-20-questions/discussion/531083

Since I joined this competition at the last week, so I just tried some simple approaches.

# result
- agent 1 score 1006
- agent 2 score 934

# Initial idea : Semantic binary search

I use gpt-4o and deepseek-chat as the policy agent, build multiple semantic trees based on the public keywords. I seems work good on the powerful models, but the challenge is the paired answer model is random ... 

Then I did some experiments on the open source models, such as llama3/3.1, qwen and their variants. but the results are depressed, the response consistency to gpt-4o and deepseek is only 70%~80%, which is likely to lead in the wrong direction after multiple rounds of questioning.

# Final approach 

## Agent Alpha

After that, I only have two days left. So I tried to borrow some ideas from commulity and came across AgentAlpha. It has a much higher win rate compared to any LLM based approaches,  but it also requires a significant number of kaggler to choose it. Any way, It only took one round to shake hand, so I think it is a good deal. Then I used deepseek-chat (much cheaper than gpt-4o) to generate 30K similar keywords based on the public keywords as the keywords pool of agent alpha.

## pre-defined questions and CoT-SC

When Agent Alpha is unavailable, I use a pre-defined quesiton seuqences first  ( from initial idea) , and then lever a CoT-SC Agent (Self-Consistency Chain-of-Thought Agent ) to generate questions.

There is still a lot of optimization that can be done, such as expanding the keywords list using WIKI or other knowledge bases, or do some filtering, and using vllm to accelerate the model for more thorough reasoning and voting. Due to the speed, there is no CoT then guessing word ...


Thanks to Kaggle for hosting this competition. It was fun, and I'm looking forward to more competitions like this in the furture.
