# 4th Place Solution

Competition: llm-20-questions
Rank: #4
Source: https://www.kaggle.com/c/llm-20-questions/discussion/531883

# Oxford Kaggle Club Solution for LLM 20 Questions Competition

##  Summary
Firstly, I would like to thank the organisers, competitors and my teammates @vassiliph, @jasperbutcher, @devnirwal01. It is clear that the difference between a 4th and 11th place finish was purely luck, and so it is with humility that we accept our prizes. We learnt a lot about working with small LLMs throughout this competition (a lot more than observing our bot in the leaderboard may suggest), and are grateful to all those who put this competition together.
We first outline our answerer and then our questioner/guesser, as well as including ideas that didn't quite make the cut in each.

## Answerer Strategy
With LLMs generally being very poor at answering linguistic type questions, our answerer would initially check against a set of regex functions. If none of these functions matched, we would then pass it onto our LLM.

### Handling predefined types of questions
  In the end, we had 22 regex functions, although most of them were near useless, and were only kept in since I saw no downside to their inclusion. 
When looking at the main pipelines for winning through linguistic questions, the two main ones were: (a) alphabetic bisection (which in hindsight was clearly the most dominant) and (b) finding the first letter of keyword and then listing possible keywords.

It was very important to be able to write patterns that were as general as possible, but able to capture the two above pipelines, since many teams asked these questions in wildly varying ways. In our final solution, it is func6, func20, func21 and goofy_func that are responsible for capturing pipeline (b) as broadly as possible, and func19 for pipeline (a). As an example, these two patterns in func19 would have been responsible for capturing some of the agent alpha questions:
<pre>question_pattern = r".*(lexicographic|alphabetic|in the dictionary).*" 
before_pattern_b = r".*(?:smaller than|before|precede|precedes)\s+['\"]+([a-zA-Z\s]+)['\"]+.*" </pre>

Overall, 10 of our wins as an answerer were via pipeline (a), and 1.5 via pipeline (b) (0.5 from only asking the first letter, not the keywords). The other 7.5 come from our LLM…

### Using LLMs
We tested many LLMs before settling on one, including Gemma2, Qwen2, Llama3, Llama3.1 and Mistral. After testing each LLM’s accuracy based on a validation of ~1300 questions, both Llama models and Mistral performed best. I had considered using majority voting among more than one model, but I assumed this would violate the memory constraints; clearly @niwatori’s team proved me wrong. Instead, I settled for Mistral-7b. The main reason for this is that Mistral ‘defaulted’ to answering no (i.e it gave many false ‘no’s compared to false ‘yes’s); this meant in pipeline (b) above, I did not have to capture whether a question was of the type “Is the keyword one of the following: …” (which I found hard to do in a general manner), but rather, I just had to capture whether the keyword was in the question or not (this is the purpose of goofy_func, name courtesy of @jasperbutcher :p).

When it came to prompt engineering, it is discussed briefly below how I attempt to replace the keyword into the guesser's question, but fail. However, few-shot learning seemed to do the trick quite well anyway. I also use 3 different prompts, and select the most common answer (or ‘yes’ if yes_count == no_count). When selecting these prompts, I created ~10 candidate prompts and then test them on my validation set (this took very long, as Mistral takes ~15 seconds per response). Many candidate prompts had 85-88% accuracy, and one had 91% accuracy. This made my job very easy, I could just select the 91% accuracy prompt, and then pick the two other prompts that are ‘most different’; which is done by selecting those that minimise the sum of the dot products of the 3 answer vectors (with ‘yes’ -> 1 and ‘no’ -> -1). In hindsight, I would have done more research on prompt engineering before actually writing the initial candidate prompts (i.e. letting Mistral explain its thought process before giving the answer).

As mentioned above, 7.5 of our wins as answerer were through the LLM (+our first win before the initial keyword reset was also LLM answerer).

### What didn’t work
When ‘places’ keywords were back in play, I corrected these keywords (e.g. iraq baghdad -> Baghdad, Iraq) and replaced these keywords into the questioners prompt before sending to the LLM (e.g. Is it… -> Is {keyword}..., …the keyword… -> …{keyword}...). In theory, the latter could also be used for ‘thing’ keywords, but the grammar becomes more complex to handle and a naïve implementation showed no difference when testing on my validation set.

Using Wikipedia articles as context in the prompt was also tested to see if it would improve performance. Unfortunately, I believe one fatal flaw in my approach was not including enough of the article. I tried using re-rankers to extract the most suitable chunk (based on the question and keyword) and I also tried including just the first few sentences. Neither of these improved my validation accuracy. @isakatsuyoshi explains his implementation of Wikipedia context brilliantly in his solution. 

Another thing I tried was to play with logits, so that even if the probability of ‘yes’ token was lower than ‘no’, as long as the probability of the ‘yes’ token was not too much lower, I would output ‘yes’ anyway. This is under the assumption that Mistral “defaults” to answering ‘no’. Unfortunately, I had no luck with this, and in fact saw a considerable decrease in my validation as a result of this.


## Guesser Strategy
We initially developed a questioning tree with keyword propagation (more on this below), but it was clear that many of the submitted LLMs were bad at answering questions. On this basis, we made a last minute switch to the Agent Alpha answering technique one day before the deadline (as a result, our final submission was only submitted with 20 minutes to spare!) We created a set of keywords (more on this below) and an associated likelihood of each keyword being selected. We then simply applied a bisection algorithm on these keywords, and for the guesser, select the most probable possible keyword from our list.

Another dilemma was whether or not to include the handshake ("Is it Agent Alpha?") or not. In the end we opted to, since we thought that losing one turn was a little price to pay in return for the possibility to ‘unlock’ a 100% correct answerer, though once again, such decisions were very last minute.

### Keywords
Our solution works with a pre-calculated list of potential keywords.
There is a dilemma of whether to use a large keyword list to maximise coverage, which will require more questions to guess, or use a smaller list of keywords that are likely included in the private keyword list, which offers smaller coverage but requires fewer questions to guess. Initially, we considered choosing one approach based on the expected meta game, but then we discovered that it is possible to combine these two approaches by adding a probability for each keyword to be in the private keyword list. So our keyword list is a dataframe with two columns: keyword and probability.

To generate the list of keywords with probabilities, we did the following:

1. Split the public keyword list 50/50 into training and validation sets
2. Categorise all the training keywords into 12 categories (e.g., Home and Living, Technology and Electronics, Hand Tools, etc.)
3. Split each category into several subcategories (e.g., Home and Living -> Furniture, Appliances, Kitchen Items, Home Decor, etc.)
4. Further divide each subcategory into third-level sub subcategories (e.g., Furniture -> Seating, Tables, Storage, Beds, Outdoor Furniture), resulting in a total of ~1700 third-level sub subcategories
5. Use an LLM to generate 100 possible keywords for each sub subcategory, using relevant training keywords as examples
6. Collect all the generated keywords into one large CSV file
7. Repeat steps #5-#6 five times
8. Count how many times each keyword was generated, assuming that a higher count indicates a higher probability of the keyword appearing in the private keyword list
9. Add the most popular English nouns with counts depending on their frequency
10. Add the list of countries and cities with low probability just in case

### What Didn't Work: Question Tree Approach
Initially, we developed a sophisticated question tree approach that we believed would be more effective than simple bisection. This method involved creating a decision tree structure where each node represented a question, and the keywords were propagated through the tree based on their answers to these questions.

Our implementation included:

1. **Tree Structure**: We created a node class to represent each node in the decision tree, containing information about the question, keywords, and child nodes.

2. **Keyword Propagation**: We used LLMs to answer questions for each keyword, determining its path through the tree. This process allowed for batch processing of keywords to improve efficiency. Importantly, if LLM answers to these questions were not consistent (we asked LLM several times), the keywords were propagated to both subtrees with corresponding probabilities.

3. **Question Generation**: We implemented a module that used LLMs to generate and evaluate potential questions for each node. It would create multiple question candidates, assess their effectiveness in splitting the keywords as information gain based on how equally the keywords are divided by this question (ideally we want 50/50 split) and how consistent are the answers. Then select the best question to add to the tree.

At the end of this process, we would have a list of keywords with probabilities of each keyword being in the corresponding subtree based on the LLM answers. This probabilistic approach allowed us to handle uncertainty in the LLM responses and potentially make more informed guesses.

The potential advantages of this approach in theory would be that these questions could be easily answered by LLM based bots. However, we ultimately decided not to use this approach in our final submission because we realised that many of the other bots' LLMs had very low quality of answers, which would significantly reduce the effectiveness of our questioning strategy.

Despite not using this approach in our final submission, developing this system provided good insights into creating these decision trees with LLMs, and given a similar competition that doesn't allow for Agent Alpha-like strategies, this is something that we would certainly explore further.

## Some thoughts
Once again, the competition was very enjoyable; although the inclusion of public keywords was certainly unfair on many bots. Once again, I would like to thank all my teammates for their work and all those that made this competition possible. I learnt a lot throughout the process and looking at all the other top competitors’ solutions!

## Code
Final Submission Notebook: https://www.kaggle.com/code/karamalrobaie/final-submission-2?scriptVersionId=195165139
Github (contains keyword generation code): https://github.com/vassiliphilippov/oxfordkaggleclub-20questions-keywords
