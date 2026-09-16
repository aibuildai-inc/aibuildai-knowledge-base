# 3rd place solution

Competition: llm-20-questions
Rank: #3
Source: https://www.kaggle.com/c/llm-20-questions/discussion/531387

I joined the competition almost exactly 3 months ago, with a specific goal in mind -- to win a prize. I admit, I have to pay my bills. Believe it or not, but with a PhD in Artificial Intelligence, I couldn't find a job for almost 3 years. Hundreds of applications just rejected by countless HR departments without any feedback or after a brief interview at best, and no, I didn't have any astronomical expectations. And so I'm not even looking anymore. I'm happy with Kaggle. And yes, I will tell this story every time I win a prize, that's going to be my revenge! xD 

Coming back, there was plenty of controversy about alphabetical bisection. For those who didn't follow, let me refer to the [post and comment](https://www.kaggle.com/competitions/llm-20-questions/discussion/511343#2866948) that brought up this approach, which finally dominated the top 20. I know many people were not happy about it, because it didn't fit the original concept of the competition. Well, that was a reality check. It was sitting there, not so difficult to spot, just waiting to be used, either openly or in cahoots. So I figured the best way to prevent any secret collusion is to bring it up openly. In retrospective, I don't know if such a collusion could really work out, but it looked like a real threat at the time.

So without any further reflections, let me describe the solution, starting with the difference between my two agents -- one was actively alpha, always offering a friendly handshake of hope for mutual understanding. The second was a shy Omega, only accepting the handshake but never extending its own. Despite lagging at first and spending time in the pit, the active alpha fared much better once it met other alphas and half-alphas. But my Omega did relatively well, too, it landed with the highest score among half- and non-alpha agents, `1081.7`. The next in line was YOLO @manh152924 with `1036.7` -- congrats buddy!

### The answerer

Is much simpler than the questioner/guesser, and it has just three components:

1. Regular expression handler
2. Alpha answerer
3. Llama 3 answerer with a simple system prompt and 6-shot user prompt

So obviously, the regular expression handler can handle all the common "first letter", "end with" and similar questions. It was just a matter of doing homework to ensure as much cooperation with asking agents as possible. It's a bit weird that some players would come up with their own "original" questions while a more common syntax was already there. A mistake in my opinion, and certainly I did the homework mainly for my agents' benefit, not theirs, but after all it didn't matter much. 

The alpha answerer is also based on regular expression matching, except it was a part of the Agent Alpha class.

When a question was not matched by any regular expression matching, it was passed to Llama 3, vanilla HuggingFace 8b-instruct. I've also tried Phi3 and Gemma 2b and 7b, but I found Llama 3 answering simple questions quite well, and it was also good in other subtasks, so I just stuck with it. At first, I struggled with too fancy answers and incorrect format, but reducing the temperature to minimum, and giving it a 6-shot user prompt was enough to ensure it sticks to "yes" or "no" in a quick timing, and quick timing was also important for testing.

One minor trick was a slightly different prompt depending on whether the question involved "keyword" or "it", e.g. "Is the keyword related to nature?" vs "Is it related to nature?". However, I didn't even try to measure whether it made any difference in answer quality.

Anyway, the answerer part was only important when playing with non-alpha agents, because for alpha search the answerer was just a trivial one-liner.

Overall, I found it a little bit tedious to work with LLMs while trying any elaborate "system of prompts". It is hard to see which parts of the prompts will be "understood" by LLMs and which will only confuse them. Perhaps the methodology can be improved in the future, perhaps it will evolve, but for now it is based on trial-and-error and quite tedious.

### The questioner and guesser

The questioning and guessing pipelines are tied together, so I will describe them together. They consist of 3 components with the following priority:

1. Alpha -- alphabetical bisection
2. Omega -- offline policy based on entropy minimization and Bayesian update
3. Llama -- simple backup

And whichever agent in that order can respond with a proper question or guess, given the context, the response is sent. 

#### Alphabetical bisection 

It is simply optimal when keywords are known and each of those keywords is equally likely to be the secret keyword during a game. Certainly there are many other equally optimal "vehicles" of bisection, it could be done with regular expressions or asking questions whether the secret keyword is "on this very long list of keywords", etc. I cannot think of any more simple and clear approach than bisection using **alphabetical order**, though. 

When the list of possible secret keywords, i.e. the private keywords, is unknown, it gets more tricky. The overall performance question shifts to the quality of our vocabulary, i.e. how many private keywords are covered and how big is our vocabulary. Moreover, it's possible to take advantage of our knowledge of the language and the fact that the private set was supposed to be "similar" to the public set.

In theory, the algorithm could work over unlimited list of keywords, with each keyword having the likelihood of being the secret one assigned. If we kept bisecting the vocabulary with respect to the sum of likelihoods (@cnumber did it!) and keep guessing only the most likely keyword from all the remaining keywords, then the performance could still be optimal -- it would depend just on the quality of our likelihoods, and with perfect assignment the whole solution would be again unquestionably optimal.  

The vocabulary in my alpha was exactly 60,008 keywords, and it included public keywords just in case -- which turned out to be the case for a while. The keywords had been collected from a few sources:

1. Llama, prompted to generate lists of similar things to each public keyword. This had the highest accuracy when tested with a validation set of public keywords, especially for the few initial keywords that Llama was generating in each query.
1. Two lists of words from previous research on the 20 Question problem:
   * Zhang Y., Lu J., Jaitly N., Probing the Multi-turn Planning Capabilities of LLMs via 20 Question Games, arXiv:2310.01468v3, 2024
   * Bertolazzi L., et al., ChatGPT’s Information Seeking Strategy: Insights from the 20-Questions Game, Proceedings of the 16th International Natural Language Generation Conference, pages 153–162, September 11–15, 2023.

2. A few public wordlists, but only with nouns and excluding proper nouns, also some lists of animals and plants.
3. A few hundred words in specific classes like "outer space objects" generated by ChatGPT.
4. Lists of consumer product categories, compiled from Amazon, Facebook and Google marketplace listings.

Compiling them into one good list was a bit messy, mainly because of plurals, but also because of some dilemmas regarding normalization, e.g. letter case, punctuation, digits and dashes, joint vs separate spelling, etc. Well, I thought removing plural/singular alternatives was quite worthwhile (especially for the Omega agent), but it wasn't entirely straightforward. I didn't find any fully reliable method for such a big dictionary of keywords, many of which were 2-word or even 3-word long. The best I found was `inflect` package. I also tried using Llama, but it wasn't good on this task at all, and it was slow. Only later, during the evaluation it occurred to me, that having only either a plural or singular form in the vocabulary was actually a mistake! For those who didn't consider it yet -- suppose the secret keywords is "shoe" but we don't have the singular form in our vocabulary, only the plural. At some point alpha may ask, "Does the keyword precede "shoes" in alphabetical order?" -- the answer is "yes" and we have just excluded "shoes" as if it was a completely different thing than "shoe". In contrast, the Omega agent described below, didn't have this problem at all and cleaning the vocabulary from plural/singular alternatives was actually beneficial. The same applies to compound vs separate spelling, e.g. "anthill" vs "ant hill".

Anyway, I ended up with 60,008 keywords of which 1468 turned out to be exact hits. 1468 out of 1798 private keywords was not bad, perhaps because the private keywords turned out simpler than I expected.

The final list of keywords actually had a form of a ranking, where each keyword had a score based on the "qualities" of its sources. The quality of a source was calculated based on how similar its list was to the list of the public keywords, and if the keyword was in multiple sources its score was adding up. So I had a beautiful ranking, which then... I forgot to apply to boost the guessing accuracy. And this is undoubtedly the **biggest flaw** of my solution, and I'm sure it would have made a big difference if it was in place. Basically instead of popping any word from the vocabulary as a guess, it should be the keyword with the highest rank, i.e. the most common and the simplest keyword available. I realized I forgot to finish this part when I saw Benjamin Kovacs's agents guessing keywords surprisingly well in the beginning. Just 2 simple lines of code were missing. @isakatsuyoshi also had a similar idea, although it was a bit coarse -- it involved two subsets of keywords, the easier and the harder. Fortunately for me, not many other alphas applied this basic trick, or at least it wasn't that apparent.

To estimate the difference in guessing performance, let's consider only those 1468 cases in which the secret keyword is in the vocabulary. With uniform guessing, the keyword is expected to be found in roughly `1 + log2(30004) = 15.87` round. If the ranking was applied properly, the average index or position of the secret keyword in the vocabulary is 6123.2 and `1 + log2(6123.2) = 13.58`. Therefore, in `1468/1798 -> 82%` of games, the secret keyword would be found in 2.3 rounds sooner on average.

And you see, some players looked down on such a simple algorithm a CS student could implement in a class. Yet why it was so easy to miss equally simple improvements and make it better? Maybe it's because it's easy only after we see it? Certainly another factor is that alpha was not expected to play such a big role in the final, and even I was more focused on other parts of the solution than alphabetical search.

#### Alphabetical search extension

Going back to the vocabulary, still, 330 keywords are missing, and that's where alphabetical search extension steps in when the vocabulary is exhausted in a game. The search ends with bounds, e.g. "tow truck" and "towel", and we ask Llama to generate more candidates within the bounds. The complication is that LLMs are not very good with letters and completions like this*. But it's possible to help them. Iteratively:

1. Ask to generate a list of words starting with the common prefix of the lower bound and the upper bound. For some prefixes it would work, for other not, but keep any words actually in the bounds.
2. While there are not enough candidates, add some likely next letters to the common prefix and ask again.
3. When we have a couple of candidates within bounds with a decent coverage of the next letters, select the middle keyword as the next testword and ask the alpha question.
4. Collect the answer and choose a guess from the right subset of the generated keywords. Ironically, here in the less important part, I didn't forget to choose the best keyword -- the choice was based on quick evaluation of candidates by Llama, scoring each candidate keyword with respect to its "commonness".

With this procedure the bounds should keep narrowing down consistently, unless of course the LLM is totally out of ideas how to complete the prefix for given next letters. The procedure is a bit tricky and was also causing some timeout troubles, but finally I managed to restrain it to safe time limits (also thanks to @matthewsfarmer suggestion), and fortunately it never errored. I was surprised by its effectiveness, I haven't measured exactly, but my rough estimate is that this extension can finish the search successfully about 40% of time, given my vocabulary and the 3 remaining guesses.

*) In his solution, @cnumber mentions specialized models that perhaps can do the whole trick by themselves.  


### Omega -- offline policy

I think a similar idea is already described in [Kha Vo's notebook](https://www.kaggle.com/code/khahuras/offline-policy-questioner-agent) notebook and also in majimekun's solution and perhaps some other, too. It involves a fixed set of questions and a fixed set of keywords with annotated "correct" answers -- i.e. we have it in our knowledge base that A is the right answer to the question Q for the keyword K. Note that A can be binary "yes" or "no", but even better it can be expressed as a degree of truth, between 0.0 and 1.0.

So let's say we have this matrix, we start with a uniform distribution of prior probabilities, expressing the probability of each keyword being the secret one. Then, the question is selected in a way as to minimize entropy over that distribution of probabilities. Miraculously the first question selected is the one which would yield as much of "bisection" as possible. The entropy of a probability distribution is quite strange and not easy to visualize or conceptualize, and so it works almost like magic.

Once the answer arrives we can update the probabilities accordingly. In a prototype I tried to update using some heuristics, like for example: increase the probability of keywords for which the answer matches, and decrease otherwise. It worked, but soon I discovered there is only one correct way of doing this -- the **Bayesian update**.

While entropy minimization takes care of selecting the best question available, the Bayesian update keeps track of the probability distribution in the best way possible. So basically after each question and answer we update the whole distribution of probabilities `P(H|E)` based on

1. the current probability `P(H)`,
2. the probability of hearing answer A -- `P(E)` and 
3. the probability of hearing answer A if the keyword is indeed correct, `P(E|H)`.

```
P(H|E) = (QR * P(H) + (1 - QR) * (1 - P(H)) * P(E|H) / P(E) 
```
It's a bit surprising how "question reliability", i.e. `QR`, found its place in the Bayes' theorem without any trouble. Note, it could be also called "answer reliability" or just "probability of hearing the truth". Perhaps it's not theoretically well-founded, or maybe there's even something wrong with it, but it worked.  

Interestingly, this extremely robust approach allows us to ask any questions in any order, without any particular semantic "narrowing down". And actually this was the main reason of going this direction -- to have an easy set of universal questions, rather than a sophisticated and labour-demanding tree of increasingly specific questions. In theory, we could ask literally any nonsensical question and still get useful information, provided:

   * our knowledge is roughly compatible with the answerer in the wild, for instance if a similar model and prompt were used on both ends,
   * the question can yield variability in answers with respect to the keywords, i.e. the question that would be answered differently for different keywords.


There are a few more ways in which this engine was further enhanced.

As mentioned, the answers in the knowledge base were a **degree of truth**, i.e. contained some uncertainty, which undoubtedly made the update smoother. So when answers were collected from Llama 3, the prompt allowed to express some degree of confidence, and so the answers could be scaled to `[0.0; 1.0]` range easily. I opted to rely on Llama solely, and then Llama 3.1 for verification, instead of some more powerful models like ChatGPT, mainly because I figured that while ChatGPT may be objectively more accurate, Llama 3 answers will have some "compatibility" advantage, as many players apparently used Llama as their answerer. 

We can easily account for "answer reliability" or **"question reliability"**, whichever is available to us. In my case, I collected answers from Llama 3 for about 95 questions and 5294 top ranked keywords, out of which 1039 were exact hits and 1079 were hits if we include plural/singular alternatives. Unlike in alphabetical search, plural/singular forms were not a problem here, and it was preferable to have just one form. Many questions were nested or conditional, though -- there's no need to ask "Is it sweet?" if the answer to "Is it edible?" is "no". But as mentioned, I didn't want to go into it too much, just a crude shallow tree with at most 1 level of nesting. 

In order to estimate question reliability I used Llama 3.1 and the real answerer prompt giving only "yes" or "no", on a subsample of keywords to reduce the effort. It turned out some questions were highly reliable, above 90%, while some other were below 80%. The reliability was simply inversely proportional to the mean absolute error between the degree of truth by Llama 3 and the plain answer of Llama 3.1. This information was used in both the question selection and the Bayesian update. So naturally questions with high reliability were preferred because they were able to bring down the entropy faster.

I have also used the question "Does the keyword start with one of the letters 'A', 'B' or 'C'?" very liberally. Actually about 15 such questions were generated for random letters in each round, and I could simply leave it to the question selection by entropy minimization to decide whether they were good or not. Perhaps those questions asked by my agents looked a bit chaotic, but it was under control. The reliability of these questions was not easy to figure out beforehand, though, so I just assumed they had reliability 0.88 at the start of the evaluation and going up to 0.95 by the end. Just a crude assumption but I think better than assuming a default 1.0 reliability rate. 


### Llama

The Omega questioner/guesser worked very well in self-play, having the guess rate around 95% when the secret keyword was in its knowledge base. But if it was missing, then it wouldn't be able to guess it at all. So here comes the rescue -- Llama 3 that would basically take all the knowledge gathered so far in a game, along with Omega's list of "similar things" and suggest a new guess. Questions could still be asked by Omega, regardless of the guesses. I also tried asking questions using Llama, but too often they were erratic, and besides, Omega couldn't use those questions to update the probabilities and suggest most likely similar things. I set 16 rounds limit for Omega, after which Llama was given the 4 remaining shots. This approach was not very effective, though, roughly only 1 in 10 games could be finished successfully by Llama in self-play with missing keywords.


### Wrap up

It was a long journey. The top spot was within reach just hours before the finish, but I don't feel much regret that it slipped out. I had plenty of luck anyway, so please let me congratulate the winner @cnumber and the runner-up @jademonk, as well as the lucky guys at 4th @jasperbutcher and 5th @yukkymaru. At the same time, I can only sympathize with other players who ended up within a single score outcome from the end zone. I know agent alpha played a role in the final decider, but surely everybody know it wasn't picking sides. Congrats to all the medalists and I'm afraid I have to stop there, as the "finishers" have already consumed enough GPU.

Many thanks to the organizers and players for this season. Thank you, @bovard, for taking action when it was necessary and only when absolutely necessary. And special thanks to @cdeotte and @khahuras for being friendly and helpful to everybody all the way. See you in another challenge soon!
