# 1st Place Solution

Competition: llm-20-questions
Rank: #1
Source: https://www.kaggle.com/c/llm-20-questions/discussion/531106

I would like to express my sincere gratitude to the organizers for creating this competition, tirelessly working to fix bugs in the matching system, and striving to ensure fairness for all competitors.

It's worth noting, however, that the leaderboard remained somewhat unstable until the end, and any competitor in the gold zone could have potentially secured the win.

Thus, I present this solution with a mix of humility and gratitude for my fortunate outcome.

# Results

- Agent 1: 1259.9 (1st place)
- Agent 2: 1228.2 (equivalent to 4th place)

# Guesser Strategy

Recognizing that top-performing Answerers were likely to implement Agent Alpha mode, I decided to incorporate it into my Guesser strategy for competitive play.

The next question was whether to ask "Is it Agent Alpha?" as the first question, or to force the Agent Alpha mode to reduce a turn.

Ultimately, I chose to ask "Is it Agent Alpha?" due to the significant advantage it provided in non-Agent Alpha matchups. This approach was necessary to recover from the "pit of dumbness" caused by the small percentage of Agent Alpha Answerers in the low score region. Additionally, losing when at the top of the leaderboard but paired with a low-scoring agent results in a substantial point loss.

In hindsight, a Force Agent Alpha approach might have been viable, given the final competition landscape.

## Agent Alpha

**Keyword List:** I created a list of potential keywords (116,937 nouns) using Python's NLTK library. Only nouns were selected as the competition specified that keywords would be "things."

**Keyword Probability Estimation:** To optimize Agent Alpha's reward expectation, I estimated the likelihood of each keyword appearing in the test data by considering:

1. Number of words in the keyword
2. English word frequency using the [English Word Frequency dataset](https://www.kaggle.com/datasets/rtatman/english-word-frequency)
3. "Thing-ness" probability estimated using GPT-4o mini

The "thing-ness" was calculated by asking GPT-4o mini the following question and then obtaining the probability that the next token would be "Yes" or "No":

"Would the word '{keyword}' generally be considered a thing?"



The scatter plot demonstrates the impact of the probability calculation. Public keywords tend to have a lower thing probability rank (meaning high probability) and a low frequency rank (meaning high frequency). The ranks and values are inverted because I ranked the keywords from high value to low value.



Using this heatmap, we can calculate the probability of each keyword being included in the private keyword list, based on the observation that the public keyword list closely resembles the private one.

In the actual calculation, a smoothing factor was introduced, but the essence remains the same.

For keywords not found in the English Word Frequency dataset (primarily compound words), only the thing-ness was used to determine the probability.

Below is a table of a few examples:

| Keyword | Probability |
| --- | --- |
| towel | 0.222 |
| titanium | 0.130 |
| chocolate egg | 0.050 |
| axis | 0.046 |
| mindset | 0.012 |
| banana boat | 0.005 |
| toolmaker | 0.002 |
| celastrus | 0.002 |
| ticktacktoo | 0.002 |

With the probability of each keyword being in the private list established, we can perform an efficient binary search that always narrows down the probability by half, rather than focusing solely on the number of keywords. This results in a more efficient search.

My first-place agent uses this biased keyword probability, while the fourth-place agent assumes equal probability for every keyword. The impact of the biased probability method is evident in the average win reward as a Guesser calculated by @lohmaa [[post](https://www.kaggle.com/competitions/llm-20-questions/discussion/529683)].

I did not come up with dynamically generated keywords that were not in the prepared keyword list, so a huge applause for those who achieved it.

## Natural Questions

I employed an entropy-based approach quite similar to the [maejimakun solution](https://www.kaggle.com/competitions/llm-20-questions/discussion/529931), selecting questions that minimized expected entropy. For the specific equations, please refer to their solution (and don't forget to upvote!).

The main difference lies in how the keywords and questions were chosen:

- The final list comprised the top ~35,000 keywords ranked by their estimated probability (as described earlier). They were actually constructed in a different way, but it seemed that it is almost equivalent to this explanation.
- Three types of questions were used:
    1. 26 alphabet-specific questions (e.g., "Does the keyword begin with the letter 'a'?")
    2. ~3,000 from winning games in the public leaderboard
    3. ~10,000 generated using GPT-4o mini

To generate questions using GPT-4o mini, I simulated matches and prompted the model to create questions that could effectively differentiate between likely keywords at each stage of the game.

For example, to distinguish between "apple," "banana," "orange," and "mango," I would ask GPT-4o:

"You are playing the 20 Question Game, and currently have to ask a question to guess the keyword. The current keyword candidates are 'apple, banana, orange, mango'. What question would narrow down the keyword candidates to half? Output only the question."

A typical response might be: "Is the fruit typically yellow when ripe?"

To construct the probability table, a strategy similar to the maejimakun team was used. Three LLMs (Meta-Llama-3-8B-Instruct, Phi-3-small-8k-instruct, and gemma-7b-it) were used to estimate p(keyword, question) for each keyword-question pair. The results from the LLMs were averaged at the end.

Each LLM was prompted with: "The keyword is {keyword}. {question} Answer the question above with 'Yes' or 'No'." The probabilities were derived from the likelihood of "Yes" and "No" being the next token.

Given the final table's massive size (approximately 35,000 x 13,000 = 455,000,000), I leveraged vllm for faster processing. Initially working with a 4090 GPU locally, I scaled up to a rented 8x RTX 4090 setup on Runpod for the final week of the competition when I realized time was running short.

This approach proved highly effective, securing a top position on the public leaderboard when there were fewer Agent Alpha Answerers at the top. The investment in computational resources (approximately $500 in server costs) was well justified by the final results.

# Answerer

For the Answerer component, I employed a dual-LLM approach, leveraging two distinct models to handle different types of questions:

1. **General-Purpose LLM:**
    - Model: meta-llama/Meta-Llama-3-8B-Instruct
    - Purpose: Handling a wide range of general questions about the keyword
2. **Specialized Mathematical LLM:**
    - Model: DeepSeek-Math
    - Background: This model gained popularity during the recent ["AI Mathematical Olympiad - Progress Prize 1" competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize), which focused on solving mathematical problems using LLMs.
    - Functionality: DeepSeek-Math takes mathematical problems as input and outputs Python programs to solve them. For simpler queries, it provides direct answers without generating code.
    - Purpose: Addressing complex, mathematically-oriented questions about the keyword

The integration of DeepSeek-Math significantly enhanced the Answerer's capability to handle intricate queries such as:

- "Does the keyword contain the letter 'a'?"
- "Does the keyword have two or more vowels?"

This specialized model allowed for high-accuracy responses to questions requiring precise letter counting or pattern recognition within the keyword.

**Model Selection Logic:**
The choice between the two models was determined by a rule-based system, primarily triggered by the presence of specific keywords like "letter" in the question. While this approach proved effective, I believe there might be room for improvement in the model selection process.

# Final Thoughts

I thoroughly enjoyed this competition, despite some challenges such as the inclusion of public keywords during the evaluation period, the slow convergence of scores, and the continuous oscillation of scores at the top of the leaderboard.

As noted by @jeannkouagou [[post](https://www.kaggle.com/competitions/llm-20-questions/discussion/530931)], the final day of the evaluation period was particularly stressful, making it difficult to sleep (the competition ended at 9:00 AM here in Japan).

I'm relieved that the competition has finally concluded, and I'm looking forward to getting some well-deserved rest.

Thank you for reading, and once again, my sincere gratitude to the hosts for organizing this competition.

# Code
The code is available [here](https://www.kaggle.com/datasets/cnumber/llm-20-questions-final-submissions). If you wish to test Agent Alpha and don't have a GPU, you may need to change `device="cuda"` to `device="cpu"` in the `utils/beta.py` file.
