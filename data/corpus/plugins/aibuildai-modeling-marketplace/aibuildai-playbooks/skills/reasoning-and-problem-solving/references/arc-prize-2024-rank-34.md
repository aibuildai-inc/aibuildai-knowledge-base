# ARC Prize 2024: Silver Medal Solution: 37th Place

Competition: arc-prize-2024
Rank: #34
Source: https://www.kaggle.com/c/arc-prize-2024/discussion/545886

Silver Medal Solution ( 37h) for "ARC Prize 2024
Create an AI capable of solving reasoning tasks it has never seen before". 
https://www.kaggle.com/competitions/arc-prize-2024/

I would like to express my heartfelt gratitude to Kaggle and the competition Hosts Abstraction and Reasoning Corpus for organizing this challenging competition. Congratulations to all the prize and medal winners in this competition. I am glad to win the Silver medal with a rank of 37tth.

This Competition generated lot of excitement with very high prize moneys and aim to rekindle interest in AGI. 
The participants came forward to share code and discussion topics. I am glad to participate in a competition involving very challenging problem set. I was able to learn a lot through watching videos of organizers, reading articles and papers, discussions and code posted by participants. 

For my solution for the ARC Prize 2024  Competition, I started with public notebooks and improvised upon  them. I used LLama 3.1 8b LLM and combined it with ARC Prize 2020 winning solutions to improve the quality and variety of responses.

1. LLaMA-generated solutions are adjusted according to specified conditions (e.g., length of prediction list) to ensure optimal responses for each task.
2. The core logic  creates solutions for specified tasks with parameters controlling token limits, sampling, and temperature settings. The function uses these settings to guide solution generation, aiming to provide diverse answers to the task at hand. Temperature setting was fine-tuned based on several iterations.
3. Conditional Solution Handling: Different conditions handle cases based on the length of prediction results. When predictions have no or very few elements, the function defaults to using LLaMA’s primary solution. As prediction list length increases, the function pulls from other  solution options, allowing more flexibility in solution selection.
4. Guidance for LLaMA’s Role: The prompt describes LLaMA  must think creatively to solve tasks. It encourages the model to generate multiple solutions and choose the best one.
Emphasis is placed on learning generalized rules to handle similar problems, pushing LLaMA to produce versatile, reusable solutions. Also, some of the prompting is based on  lateral thinking techniques.
5. Integration of Outputs: The function includes logic for merging LLaMA’s output with other models' solutions( ARC 2020), by incorporating fallback and redundancy mechanisms.
6. This structure allows LLaMA to take on abstract reasoning challenges, combining its own outputs with a secondary model to improve the quality of responses across diverse reasoning tasks.

The final Leaderboard score = 27 

References:
1. Francois Chollet, Mike Knoop, Bryan Landers, Greg Kamradt, Hansueli Jud, Walter Reade, and Addison Howard. ARC Prize 2024. https://kaggle.com/competitions/arc-prize-2024, 2024. Kaggle.
2. ARC24 Developed 2020 Winning Solutions, https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions
3. https://www.kaggle.com/code/yogi050294/04-inferencing-llama-3-1-8b-unsloth
4. https://www.kaggle.com/code/feiwenxuan/arc-prize-2024-10-14
5. CRS Kumar, "Towards Artificial General Intelligence: Enhancing LLMs capability for Abstraction and Reasoning"DOI: https://doi.org/10.31224/3863, https://engrxiv.org/preprint/view/3863
