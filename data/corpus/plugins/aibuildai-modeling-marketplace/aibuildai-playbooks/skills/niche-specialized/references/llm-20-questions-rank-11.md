# 11th place solution (majimekun)

Competition: llm-20-questions
Rank: #11
Source: https://www.kaggle.com/c/llm-20-questions/discussion/529931

This is majimekun team’s solution. majimekun is composed of @niwatori, @akimaru, and @keisho.

This competition was uniquely interesting. I would like to express my gratitude to the organizers. However, we believe the competition might have been even more engaging without the binary search component, allowing for more diverse strategies and creative problem solving.

# Summary
Our solution is primarily classified as an offline policy approach. The main motivation behind adopting the offline policy is that while an LLM with 7~9B parameters may struggle to make accurate guesses based on 20 question-answer pairs, it performs well when answering a single keyword and question pair.

We prepared a keyword list, a question list, a probability table for all the (keyword, question) pairs, and prior probabilities of keywords beforehand. During inference, we derive an optimal question by minimizing conditional entropy, and guess a keyword with maximum likelihood. Our keyword list was generated using LLMs, our question list was obtained from public replays and LLMs, and the probability table was calculated from LLMs.

According to [the nice statistics](https://www.kaggle.com/competitions/llm-20-questions/discussion/529683) by @lohmaa, the win ratios of our pure LLM Guessers during the evaluation period are (it should be noted that the win probability depends on paired answerers)
* submission_id=39526145 (K=10): 3/14 (21.4%)
* submission_id=39527069 (K=400): 5/40 (12.5%)

, where K is a parameter that limits the ratio between maximum and minimum prior probabilities of keywords.

# Answerer
The Answerer component is relatively straightforward. It consists of a few rule-based logics and majority voting among three LLMs.

## Rulebases

The Answerer responds if the question matches the following regular expressions:

* startswith and endswith: `(starts|start|begins|begin|end|ends) with the letter [\'"]?([a-zA-Z])[\'"]?\?$`
* contains: `(includes|include|contains|contain) the letter [\'"]?([a-zA-Z])[\'"]?\?$`
* dictionary order: `keyword.*(?:come before|precede) \"([^\"]+)\" .+ order\?$`

The dictionary order regex captures the Agent Alpha type questions as well.

## LLMs

If the question cannot be answered by the rulebases, we rely on majority voting of three 4-bit quantized LLMs: gemma-9B, llama3.1-8B, and mistral-7B.

We assumed that majority voting would be more robust than using a single 8-bit model in such a noisy environment, so we opted to use the three models. Initially, four models, including phi-3-small, were considered, but after a rough simulation, we selected the final three models.

Since loading more than two models simultaneously exceeds the GPU memory limit, we load and unload models for each inference. We reduced the number of model loads per turn from three to two by rotating the inference order like ABC/CAB/BCA/…, where A, B, and C represent each model.

Initially, we used the probabilities of model outputs obtained by summing up contributions from representing tokens. However, this method performed worse than the simple majority voting based on text generation. It’s possible that some tokens were missing in the probability calculation in our implementation.

# Guesser

Our Guesser is based on the offline policy. We prepared a keyword list, a question list, a probability table for keyword and question pairs, and prior probabilities of each keyword.

Deriving the optimal question and the optimal keyword follows the straightforward logic. Given a list of known question-answer pairs, the optimal question is determined by minimizing the conditional entropy:

$$argmin_{q} \sum_{z=Yes,No}p(q=z|q_1,...,q_t)[-\sum_k p(k|q_1,...,q_t, q=z) log p(k|q_1,...,q_t, q=z)],$$

where \\(k\\) represents a keyword, \\(q\\) represents a question, and \\(q_1,...,q_t\\) are the already asked questions (and answers). This is calculated using Bayes theorem, complexity of the numerical calculation is O(KQ), where K is the number of keywords and Q is the number of questions.

The optimal keyword can be derived by

$$argmax_k p(k|q_1,...,q_t).$$

We also incorporated the prior probabilities for the keyword \\(p(k)\\). The logic of deriving the prior probabilities is described in the next section.

## Keyword list generation

A keyword list is crucial in an offline policy since we cannot guess a keyword that isn’t in our list. However, the more keywords we have, the harder it is to find the right answer. Poor keywords can negatively affect both question selection and actual guesses. Therefore, we have to find a good balance of recall and precision to some extent.

To evaluate the quality of a keyword list, we randomly separated the public keywords into a training keyword list and a validation keyword list. We generated a keyword list using only the training keywords, and evaluated recall and precision (or num of vocab) with the validation list.

We experimented with various methods to generate the keyword list:

* Using all descendants of `physical_entity.n.01` in the WordNet
  * Recal: ~46%, Vocab size: ~35000
* Using neighbors of training keywords in terms of WordNet (e.g., children of parents of each keyword)

Ultimately, we found that generating keywords with LLMs (ChatGPT and Gemini) produced surprisingly better quality. We hypothesize that the Kaggle host also used LLMs to generate their keyword list. By providing a subset of training keywords as few-shot examples, we collected keywords by repeating the process multiple times.

Another advantage of using LLMs is that we can estimate the likelihood of appearance in the public/private keyword list based on how often the keyword is generated by the LLMs. We found a clear correlation between precision and frequency. Leveraging this correlation, we defined prior probabilities based on these frequencies, and included them into the optimal question and guess calculations. Thanks to the prior probabilities, we can effectively improve precision while maintaining the number of vocabulary.

The figure below shows the number of keywords and recall for the validation keyword list, as the frequency threshold changes. The right most points correspond to frequency\>=2. After a rough local simulation, we choose 2 as the threshold.




After removing clearly non-object keywords with the help of LLMs, our final keyword list contained approximately 32,000 entries.

## Question list generation

Since the optimal question is automatically derived, having more questions improves the Guesser’s performance. The bottleneck was the computational resource to calculate the p(keyword, question) probabilities table.

We prepared questions from:

* Rulebases such as “Does the word start with ‘a’?’
* Questions collected from the public episodes
* Questions generated by LLMs

The final question list contained approximately 1,000 questions.

One regret is that we should have added more questions. According to the replays, our bots sometimes end up asking many “startswith” questions, possibly because the questions are too coarse and there are no more useful questions.

## Probability table labeling

We need p(question="yes"|keyword), which is the probability of “yes” answer for each keyword and question pair.
We assumed that many Answerers in the environment use LLMs, and on average, p(keyword, question) calculated locally with LLMs would not differ significantly from the actual environment. Therefore, we used the average probabilities of four LLMs: gemma-9B, llama3.1-8B, mistral-7B, and phi-3-small. We derived the “yes” probabilities by summing up token probabilities for “yes”, “right”, “correct” and similar terms.
We used vLLM to speed up the inference for all models except for gemma-9B (because of OOM or some reasons). This process required decent amount of computational resources.

# Agent Alpha

We include the Agent Alpha capability in both of 2 bots. The vocabulary is a simple combination of words from WordNet and the keyword list used in the Guesser. The vocabulary size is approximately 173,000.

One regret is that since we could predict that there would be many bots with Agent Alpha capability, and the number of steps to find answers strongly affects the win rate in matches between binary search bots, we should have improved the Agent Alpha part more carefully. This could have included choosing the keyword list more thoughtfully, or prioritizie likely words based on the prior probabilities as discussed above.
