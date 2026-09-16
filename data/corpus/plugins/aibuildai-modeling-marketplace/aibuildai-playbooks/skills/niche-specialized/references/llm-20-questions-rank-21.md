# [21st]: Back to basics

Competition: llm-20-questions
Rank: #21
Source: https://www.kaggle.com/c/llm-20-questions/discussion/531104

# Solution writeup
First of all thanks everyone for a fun competition. Even though there were lots of issues hopefully everyone learned something. 

I'm not really sure what my true rank is as my other nearly identical bot was stuck in the pit for most of the competition and reached an elo of **749**. The bot was identical other than having one extra question: *"Is the keyword a single word?"*. This question was not very helpful based on the game histories, however the bot did not seem that much worse when looking at the replays. 

Nevertheless, I decided to finally contribute and present my solution after a competition. I have always learned a lot from other peoples' write-ups so hopefully someone finds this useful.

## Personal learning goals for the competition
I think setting learning goals is quite important when joining any competition. This way one always benefits from joining a competition even if the competition has issues (hacks, bad metrics, unfair scoring etc.) or real-life obligations prevent you from committing enough time to achieve a good rank. This learning mindset has made my Kaggle experience much more enjoyable as you stop worrying about your ranking. 

For this competition I wanted to get more practical experience working with LLM:s, as I had not tinkered with them as much I'd liked. My concrete goals when starting were:
- Get familiar with various open source LLM libs and tools
- Learn prompt engineering fundamentals and intuition
- Try finetuning if there's time 

I think I achieved these goals quite well during the competition. 

## Solution 
You can find my code in the following repo:
https://github.com/JoonasMaanonen/kaggle-llm-20-questions

I worked mostly locally and in a Google cloud instance for the GPU access. Personally, working inside Kaggle kernels feels a bit slow as I don't have my editor, git or vim bindings. I still use Kaggle kernels sometimes depending on the competition. 
### Answerer
My answerer bot used the [basic instruction finetuned LLama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).  I started with the smallest model I could find (Phi-3) as my first goal was to just get a working submission and learn the outlines lib. Quite quickly I switched to Llama3 based on quick research and browsing through the other kernels. 

I only did prompt engineering for the answerer and it worked quite well. First I played around with few-shot prompting but did not find any consistency with that. This lead me to believe that finetuning might not be best use of my time in this competition as there's lots of work involved in it and I had limited time. 

What really improved consistency of my answerer was using chain of thought prompting. Basically, just telling the model to *"Think step by step"*. Before using chain of thought prompting the model had trouble being consistent even in my simplest assertions. However, after getting chain of thought to work it passed 90% of my assertions very consistently. 

I think my biggest learning here was the benefit of creating a quick evaluation suite to assess the model quality. I just used pytest and created many test cases like this:

```python
assert (
    self.model(get_observation("flashlight", questions=["Is the thing electronic?"]))
            == "yes"
        )
```
    
This approach allowed me to iterate on the prompt quickly and assess the impact of changes instantly without having to run full games. I used this approach for developing the guesser and the answerer prompts quite successfully. However, it was not trivial to evaluate the questioner quality with this approach so there I resorted more towards actual game play and vibe checks. 

Initially, most of my guessing tests were based on the places category as those were easier to construct. Unfortunately, I had to scrap those after the places category was removed. 

## Guesser and Questioner
These bots also used the same LLama 3 model as the answerer. I tried Llama3.1 also,  but felt like it was pretty much the same performance or even slightly worse so I decided to ditch it to keep things simple. 

I started with pure LLM and used pretty much all the basic prompting techniques that were described above. After examining lots of games, I felt that the model was decent at finding out the keyword after it had determined the general category of the thing. However, the path to this general category was often very inefficient and the model kept sometimes asking questions with very little information gain. I also felt that the model sometimes followed the game history more than the instructions in the prompt, especially when the game history grew in size.

All this lead me to conclude that having a high quality game history was crucial for good guesses and questions. The first low effort idea that I came up with was to just fix the first 5-10 questions and guesses with the type of questions I think that model should ask. So I decided to try that. 
### Question and guess tree 
I fixed the first ~5-10 game rounds by making a tree of questions and guesses. I first tried a bit with a basic if else, but the code quickly got out of hand. This is where I  decided to build a basic tree from your intro to data structures class. 
The basic idea of the approach is to just recursively traverse the question/guess tree based on the answers until we hit an empty node. After reaching an empty node the LLM carries on from there and it should have a solid game history to start with. Here's a little code snippet on how the tree looked:
```python
question_tree_root = Node(
    question="Is it a living thing?",
    yes_branch=living_things,
    no_branch=Node(
        question="Is it a type of food or drink?",
        yes_branch=edible_things,
        no_branch=basic_man_made_object,
        guess="chisel",
    ),
    guess="chainsaw",
)
```
I constructed the tree based on basic intuition and added more nodes based on the failure cases observed from self play and the game replays. 

This tree approach made the final LLM guesses and questions much more consistent as the game history always started with a set of sensible questions and guesses. I know that people have used RAG to generate dynamic few shot examples for prompts with success, so I guess this sort of has echoes of that idea without the RAG part.

Additionally, the tree generated some "free" wins, where the correct guess was achieved just by traversing the tree without ever even calling the LLM. I tried keeping the questions clear and simple to understand even for the worse answerers. I also, started the tree with living things and food stuff where the good questions are quite straightforward. I felt that the various man-made objects were super hard to nail down even for human bots. This is why I tried to focus on winning the easier categories quicker. 

## Other random things/learnings 
- Sometimes the model made duplicate guesses, so I added a basic retry mechanism to the guesser if the guess was already found in the guess list. This did not prevent duplicate guesses completely and I'm not sure how impactful this change was in the grand scheme of things.  
- I also did some basic post processing like splitting on the token " or " and taking the first list element, since sometimes the model answered with two options.
- Making a complicated and long prompts with detailed game strategies did not work very well for me. As the game history grew larger the model seemed to forget the instructions and focus more on the context in the game history. I guess the smaller models have a hard time keeping everything in their context, at least that was my experience. This was a bit surprising after being used to ChatGPT4 grade models. 
- I also tried splitting the prompts into multiple llm calls. For instance, first llm generating guess candidates and then second llm reflecting and ranking those. I think the idea has promise but I kind of ran out of time and ditched this research direction quite quickly.  
- I did not pay much attention on what other people were doing in the competition, but it was cool to see all the creative alpha agent solutions. I think my agent lost most games it faced against the alpha agents. 

## Learning materials I found useful during the competition

-  Nice post about various tips for building with LLM:s https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/
- Prompting fundamentals: https://eugeneyan.com/writing/prompting/
- Evals fundamentals: https://hamel.dev/blog/posts/evals/
- Great videos about various topics on building with LLM:s https://parlance-labs.com/education/
- Outlines paper for the structured generation: https://arxiv.org/pdf/2307.09702
- Also shoutout to some kernels that helped me during the comp:
	- https://www.kaggle.com/code/robikscube/intro-to-rigging-for-llm-20-questions-llama3
	- https://www.kaggle.com/code/cdeotte/starter-code-for-llama-8b-llm-lb-0-750
	- https://www.kaggle.com/code/khahuras/offline-policy-questioner-agent

### Conclusion 
Overall, I'm happy that I participated in the competition.  It was unfortunate that the ranking probably does not accurately reflect the true skill order of the bots and all the changes during the competition were annoying. However, that is something we have limited control over as competitors. In the end, I had fun and learned a lot.

With kind regards 
Joonas
