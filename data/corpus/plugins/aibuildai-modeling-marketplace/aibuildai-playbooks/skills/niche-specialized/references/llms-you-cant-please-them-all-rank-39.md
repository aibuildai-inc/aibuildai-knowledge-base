# 39th solution

Competition: llms-you-cant-please-them-all
Rank: #39
Source: https://www.kaggle.com/c/llms-you-cant-please-them-all/discussion/566376

See my final notebook here: https://www.kaggle.com/code/jiprud/42th-27-085-essays

My submission was based on my Simple Submission notebook: https://www.kaggle.com/code/jiprud/essays-simple-submission

The path from my public notebook to the final one was as follows:

- I added a third attack to target each model separately.
- After countless attempts to find the perfect attack by tuning the prompt, I turned my attention to the list of random words.
- There were three main categories of words that showed bias (i.e. made scoring unstable): positive, negative, and writing-related words.
- In addition to these categories, my tests showed that each model (and each prompt) was sensitive to some specific words that did not fit into a particular category.
- I eliminated all these words from the original list, making my prompts more stable (albeit not 100%).
- Besides cleaning the word list, I used specific word sets in my prompts to exploit their bias potential.
- This wordplay led me to my final improvement: for one of the attacks (in the notebook I call it choices_2), I generated a list of the best-performing complete prompts from my experiments and then used them for submission.


Thanks to everyone who participated and shared their ideas.
This is my first Kaggle medal, and I have to say it was a very enjoyable and interesting experience for me!
