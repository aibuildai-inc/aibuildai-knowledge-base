# 5th Place Solution

Competition: llm-20-questions
Rank: #5
Source: https://www.kaggle.com/c/llm-20-questions/discussion/531128

First and foremost, I would like to express my gratitude to all participants, organizers, and everyone involved.

## Code

[here](https://github.com/yukimaru77/llm-20q-5th-opensource-repository)

## **Summary**  
Objects in this world have two aspects: the aspect of a name and the aspect of the attributes the object contains. When searching for a keyword, if all objects are assumed to be equally likely to be chosen, a binary search tree minimizes the expected number of operations. However, if the probabilities differ, a Huffman search tree minimizes the expected number of operations. However, since it is nearly impossible to perform a binary search on the keyword space based on the attributes contained by the objects, I focused solely on the aspect of the name in this strategy. I also assumed the probability of a word being chosen as word_frequency. **Therefore, for the questioning strategy, I used a binary search tree ordered alphabetically for the names of the objects in the early stages, and a Huffman tree-like method based on word_frequency in the later stages. For the answering strategy, I employed painstakingly hard-coded responses, as well as VAGOsolutions' Llama 3.1 SauerkrautLM 8b Instruct model. For the inference strategy, I used word_frequency to efficiently infer the keyword.**

I will introduce my approach in three parts below:

1. Responder Strategy
2. Guesser Strategy
3. Questioner Strategy

The order might seem a bit unnatural, but please allow it for the sake of explanation.

## Responder Strategy
Naturally, the most important aspect of the responder strategy is to provide the correct answer to a question. However, this is currently not feasible with small LLMs. In other words, providing the correct answer essentially means minimizing the use of LLMs and instead hardcoding responses whenever possible. However, it is not feasible to cover all possible questions with hardcoding. Therefore, I obtained all questions containing characters like ", ', and , from the public board, performed all possible pattern splits on them, and used regular expressions for matching. Questions that include characters like ", ', and , often pertain to the structural attributes of the keyword rather than its properties. Below are some examples of standard questions:

* "does the secret keyword start with a letter between 'm' and 'r', and end with a letter from 'n' to 't'?",
* "Does the first letter of the word come between 'r' and 't' in the alphabet?",
* "does the keyword start with any letter alphabetically between 'c' and 'k'?",
* "Is the second character of the keyword one of these: 'r, o, a'? ",
* "Is the third character of the keyword one of these: 'm, r, c, n'? ",
* "Is the keyword one of these: 'chair','laptop'",
* "Is the KEYWORD one of the following? 'Gadget' , 'Garage Door' , 'Garbage bag' , 'Garbage Can' , 'Garbage Disposal' , 'Garbage Truck'? ",
* "any(letter in str(obs.keyword).lower() for letter in list('q', 'z', 'x', 'j', 'k', 'v', 'w')) #Does the keyword contain any of the following letters: [q', 'z', 'x', 'j', 'k', 'v', 'w]? ",
* "does the second word contain the letter 't'?",
* "Is any of the letters ['c','v'] inside the spelling of the keyword?",
* """Does the keyword (in lowercase) precede "gus'-khrustal'ny russia" in alphabetical order?""",
* "If the keyword is lexicographically smaller than 'acapulco mexico' answer yes, otherwise answer no. INFO: Lexicographical order is the order in which words are listed in a dictionary.",
* "Is the word (in lowercase) lexicographically smaller than 'ty'?",
* "Is the word lexicographically smaller than 'zzt'?",
* "does the secret keyword start with a letter between 'm' and 'n', and end with a letter from 'a' to 't'?",
* "Does the keyword end with the letter 'e'?",
* "does the keyword include the letter c?",
* 'Sure, please ask your question: Does the keyword include the word "apple"?'

For all these questions that pertain to the structure of letters, I manually created functions using regular expressions to match and provide a yes or no answer. For keyword matching questions (e.g., "Is the keyword one of these: 'chair','laptop'"), I used the compare_words function to align with the game's specifications. This was a very labor-intensive task, but I got through it with a heart full of compassion for the other participants. For all other questions, I am using this 8B model based on Llama 3.1, which has a high iFEVal score on the open leaderboard: https://huggingface.co/VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct.

## Guesser Strategy
My agent is a binary search type alpha agent. Then, from the list of keywords, which word should we guess as the secret keyword? Here, I hypothesized that keywords with a high word frequency are more likely to be the secret keyword. Although I would have liked to verify this hypothesis by gathering data from the public board, due to time constraints, I decided to trust this hypothesis blindly. I calculated this metric for all keywords and adopted a strategy of guessing the keyword with the highest score as the secret keyword.

## Questioner Strategy
Since the list of keywords consisted only of objects that are featured, I decided to collect only objects. By looking at keyword.py, it was clear that most keywords represented tangible entities in the real world, so I focused on collecting such objects. To achieve this, I extracted only nouns from image caption datasets (Conceptual Captions, Coco Caption) and Amazon review datasets. Then, I filtered the necessary nouns by processing 100 words at a time with Gemini Flash. Although there were cases where nouns were output by hallucination, where nouns not in the original word list were still output after filtering with Gemini Flash, I added these nouns to the list, interpreting that Gemini Flash considered them important in some sense. Additionally, I added nouns from WordNet to the list.

As I briefly mentioned earlier, my agent is a binary search type alpha agent. However, strongly believing in the hardcoding approach I discussed in the "Responder Strategy," I predicted that many agents would use hardcoding for the responder strategy in this competition (although due to time constraints, this remains a prediction). Therefore, I decided to perform binary searches based on lexicographical order on all agents without an initial handshake. The reason for choosing a binary search based on lexicographical order is that it seemed the most likely to have been hardcoded. Indeed, this approach was supported by two well-known public notebooks. However, when the remaining number of keywords became small, I felt that a binary search based on lexicographical order became inefficient. So, when the total length of all keywords became approximately 1400 characters or less, I switched from a binary search based on lexicographical order to a binary search based on word frequency. Specifically, I changed the questions to the following:

"Is the keyword one of the following? {top1 word frequency's word},  {top2 word frequency's word}, {top3 word frequency's word}, …"

This allowed for a binary search using keywords with high word frequency. This idea is based on the realization that when using a binary search type alpha agent, this game is equivalent to a coding problem where bit strings are assigned to keywords. Furthermore, by treating the word frequency as the probability of occurrence, Huffman coding can be applied. This idea was implemented accordingly.
