# 2nd Place Solution for the LLM 20 Questions Competition

Competition: llm-20-questions
Rank: #2
Source: https://www.kaggle.com/c/llm-20-questions/discussion/529643

# Tricksy Hobbitses solution

>"What have I got in my pocket?" Bilbo said aloud. He was talking to himself, but Gollum thought it was a riddle, and he was frightfully upset.
"S-s-s-s-s," hissed Gollum. "It must give us three guesseses, my precious, three guesseses." <br>-- The Hobbit

## Context

Business context: https://www.kaggle.com/competitions/llm-20-questions
Data context: https://www.kaggle.com/competitions/llm-20-questions/data

## Overview of the Approach:

* Dual alphabetic binary search ("alpha") plus pure online LLM strategy
* Unigram-based alpha agent focused on long tail of single keywords with LLM extension to find noun phrases
* Use of English language frequency table to guess high probability words faster
* Auto-summarization of knowledge to preserve and consolidate context by the LLM guesser
* 100% online LLM asker/guesser with no preprocessed or offline questions
* Multi-modal question types to gain different information tags for the keyword: category, location, size, and enumerate+split
* Substituting subject with the keyword when answering

### Process flowchart



### Performance evaluation

* #2 on Private LB (alpha + LLM)
* Peaked at #7 in Public LB (LLM only)
* 89% keyword find rate in Private LB matches (alpha only) - see [analysis](https://www.kaggle.com/competitions/llm-20-questions/discussion/530312)
* 5.81 average reward for wins achieved (alpha only) - see [analysis](https://www.kaggle.com/competitions/llm-20-questions/discussion/530312)
* 12% win rate as an LLM asker/guesser during competitive gameplay
* 11% win rate as an LLM answerer during competitive gameplay
* 7th best team results in LLM-only matches according to custom metrics described [here](https://www.kaggle.com/competitions/llm-20-questions/discussion/530442)

### Source code

Full code available [here](https://www.kaggle.com/code/jademonk/tricksy-hobbitses-llm-20-questions-solution)

### Acknowledgements:

Many thanks to all competitors and organizers of the competition, but most especially:
* Kaoutar @wouldyoujustfocus and Chris Deotte @cdeotte for their starter notebooks [here](https://www.kaggle.com/code/wouldyoujustfocus/llama3-8b-llm20-questions-lb-750) and [here](https://www.kaggle.com/code/cdeotte/starter-code-for-llama-8b-llm-lb-0-750) which undoubtedly saved me hours of frustration trying to get an initial agent working in the Kaggle environment
* Matthew S Farmer @matthewsfarmer for [discovering](https://www.kaggle.com/competitions/llm-20-questions/discussion/523619) a hack to get Llama 3.1 to work on Kaggle
* Loh-maa @lohmaa for completely changing the playing field

### Details of the submission:
My solution entails both an LLM-augmented alpha-based binary search and an LLM-heavy agent with no offline preprocessing.

I began this competition attempting to build an LLM-based solution for the “things” category, seeking to tackle the toughest part of the challenge first and then build upon this for “places,” which ended up becoming irrelevant. I built the alpha binary search quite late in the process once it became apparent that there would be a cluster of high win-rate alpha agents at the top of the final leaderboard (and this would be necessary to remain competitive).

As I mention in my comment [here](https://www.kaggle.com/competitions/llm-20-questions/discussion/503429#2973042), even a small number of agents adopting this strategy could be enough to significantly change the competitive dynamic.

For LLM, I use Llama 3.1 8B 8-bit quantized with no finetuning, which slightly outperformed Llama 3.0 in my tests.

### Alpha asker/guesser
I adopted the “Is it Agent Alpha?” handshake shared in @lohmaa’s public notebook. This seemed the most direct way to determine if an answerer was equipped to handle alpha binary search questions. I figured that many top answerers would be programmed to handle this setup.

For my keyword database, I decided to use single-word (unigram) keywords only. I used this dataset: https://www.kaggle.com/datasets/rtatman/english-word-frequency as my starting point, narrowing 333,333 unigram entries down to 120k through use of two LLM queries: “is it a valid English word which is not an acronym or abbreviation?” And “Is it familiar to a layman?” I attempted to use Llama to do part-of-speech tagging along with various other filters and found it to be highly unreliable. Consequently, my 120k dataset remains full of junk words that I didn’t dedicate time to filtering out further.

My search algorithm attempts to find the first word of the keyword before attempting any phrases. During the initial search, it will use the highest-frequency unused word within each valid range as its guess (to try and get the word before other alpha agents). 

Once the first word is definitively identified (or all possible single words have been excluded from consideration), the algorithm moves on to stage two. In this stage, I use an LLM to suggest possible noun phrases beginning with the first identified word. In the situation where the first word is fairly generic (e.g. “metal”, “electric”, etc.) this strategy is likely to fail. However, many of the first words lead to very meaningful LLM-prompted phrases (e.g., my agent was able to predict “fabric glue” from the starting word of “fabric.” Highly specific terms such as “guinea” from “guinea pig” would undoubtedly also result in a win.) 

I did tests on public keywords to determine whether to truncate my unigram list to 65k, 32k, etc. to allow time for more noun phrase guessing but found that keeping the long tail of 120k unigrams provided best performance. Since the private keyword list seems weighted towards single words, I think this design choice provides very good coverage of the keywords in the first stage. 

### Answerer

My answerer operates in two stages:

**Stage 1:** Looking for alphabetical order, list checking, or letter-containing type of questions. In this stage I look for common prompts which involve word spelling or list checking, since the LLMs tend to be poor at answering these question types. I then use the LLM to extract the comparison word, letters, or list from the prompt and compute a manual comparison.

**Stage 2:** I use the LLM to answer the question. A **very important finding**[](url) is that the LLM’s accuracy vastly improves if the keyword is substituted back into the prompt before attempting the answer. So, my algorithm first uses the LLM to determine the subject of the sentence and then substitutes the keyword in place of the subject. Then it answers the question.

My performance metrics suggest that my answerer is among the top 4% of answerers for non-alpha episodes.

### Online LLM asker

My LLM does not use any offline or pre-prepared questions. I built an offline asker bot at one point but determined that my online LLM was better, so my final submissions were using my online bot only.
The LLM operates in four different modalities, which are processed in order:

1. Category
2. Location
3. Size
4. Enumerate+Split

##### Category modality


During category mode (typically the first 12 questions), the LLM seeks to subdivide its current “category” into two separate non-overlapping spaces, beginning with “things that are tangible objects.” For this, I use the following prompt:

`Divide the category “[prior category] that are [category]" into two broad, clearly-defined, non-overlapping sub-categories, ensuring that all [prior category] that are [category] fall into one category or the other, but not both. Respond with the names of the two sub-categories separated by a comma only. Do NOT repeat the original category. Use common phrasing. Each sub-category must be a noun or noun phrase.`

For this prompt, category refers to the most specific known category, and prior category refers to the category found one step prior.

The prompt results in two subcategory candidates. I test both in turn. If one returns a ‘yes’ response, the algorithm adopts this as its current category and continues to drill down. If neither returns a ‘yes,’ the same process is repeated with two more subcategory candidates. After four consecutive failures or 12 total questions, the algorithm moves on to the next modality.

I require positive confirmation before moving forward and am careful not to assume that a negative response implies the opposite is true, since in reality my category splits may not cover 100% of possible instances.

As an example, for the keyword “fox,” this approach would often determine that the keyword is something like: “a terrestrial mammal that is a predator” or “a predator that is a forest stalker”.

##### Location modality

In location modality, I enumerate possible locations where the keyword may be found, using the following prompt:

`Provide several broad categories of locations where [prior category] that are [category] are most often located. Respond with the category names in comma-separated format only.`

Then, the algorithm uses up to 5 questions to ask whether the keyword is found in each of the enumerated locations. Once a positive response is returned, this location is stored for future reference and the algorithm moves on to its size modality.

##### Size modality

During the “size” modality, the LLM essentially rephrases the two questions: Is it small? Is it large?  Hopefully one of these two will return positively.

##### Enumerate+Split modality

Once category, location, and size modalities have all been exhausted, the LLM falls back into its enumerate+split modality by default. This is actually what I coded as my first entry into the competition and remains reasonably good at asking smart questions. It operates in two stages: enumerating possible choices, then creating a question to split the choices.

In the enumeration stage I use the following prompt:

`Create a list of 30 [prior category] that are [large/small] [category] typically located in or at [location], matching the following description: [summary – see below]. The things must be unique, diverse, [category items] and as different as possible from each other. The [category items] must be common examples of [category items] and representative of a range of different possible options. The things must represent examples of as many different categories of [category items] as possible. The answer should be returned as a comma-separated list with no additional verbose output. None of the [category items] may be repeated. Each item in the list should be as different as possible from the prior item. No words in the list may be repeated. Respond with the comma-separated list only. Order the responses according to the most likely or common choices according to the criteria above.`

The summary referred to above is maintained in the guesser code, which I will describe in more detail momentarily.

This prompt tends to generate a reasonably good list of 30 or so representative examples of things that could possibly be the keyword. Then I try to find a question which will split the list into two equal halves, as follows:

`Create a simple yes-or-no question, responding with the question only and no introduction or additional verbose details. The question should broadly categorize and divide the following into two equally-sized lists: [output from the prior prompt] Do not include questions similar, equivalent, or directly opposite to the following: [all prior questions] Ensure that the question is simple, unambiguous, clear, and can be answered either yes or no. Do not create compound questions. The question may explore different aspects or characteristics of the list items including size, appearance, function, location, usage, and other defining characteristics. The question should create a general or broad classification of the two categories and should not be overly specific.`

Sometimes this prompt results in a very ingenious query. Other times it asks fairly trivial, repetitive, or abstract questions. So it’s hit or miss. But if my prior modalities worked fairly well and I have a specific category, location, and size to work with, this default modality can often ask intelligent questions to further narrow down the choice set. Or, at the very least, I stall for time to attempt more guesses within my known category/location/size.

### Online guesser

My guesser keeps a running summary of its understanding of the keyword in paragraph form. After each question and answer it will update the summary to reflect the new information gained (and discard redundant or conflicting data). I found that when feeding an LLM a nice, natural language description of the object it did a better job than having it try to parse through a bunch of yes/no questions every single round. This summary is used both by the guesser as well as the “enumerate+split” modality described above.

I found that this auto-summarization often allowed my guesser to recover from incorrect answers provided by unreliable answerers, in effect providing greater attention to consistent information.

When choosing an answer, the guesser uses the first prompt from the default modality described above to enumerate 30 possible choices for what the keyword would be. After excluding guesses that were already made (or in the public keyword list), my algorithm simply guesses the first presented choice, since that’s what the LLM thinks is the most likely answer.

##Sources

Zoe Mongan, Luke Sernau, Will Lifferth, Bovard Doerschuk-Tiberi, Ryan Holbrook, Will Cukierski, Addison Howard. (2024). LLM 20 Questions. Kaggle. https://kaggle.com/competitions/llm-20-questions

Kaoutar @wouldyoujustfocus [starter notebook](https://www.kaggle.com/code/wouldyoujustfocus/llama3-8b-llm20-questions-lb-750)

Chris Deotte @cdeotte [starter notebook](https://www.kaggle.com/code/cdeotte/starter-code-for-llama-8b-llm-lb-0-750)

Loh-maa @lohmaa [Agent Alpha](https://www.kaggle.com/code/lohmaa/llm20-agent-alpha) partial solution

Matthew S Farmer @matthewsfarmer [hack](https://www.kaggle.com/competitions/llm-20-questions/discussion/523619) to get Llama 3.1 to work in Kaggle environment
