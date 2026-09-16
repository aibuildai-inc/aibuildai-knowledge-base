# Summary of 21 solutions (LB: 0.894)

Competition: ai-village-ctf
Rank: #20
Source: https://www.kaggle.com/c/ai-village-ctf/discussion/351804

Hello everyone,

First, I would like to thank the hosts for organizing such a great (and addicting!) competition and bearing with us all this time at Discord! It was certainly a very unique and educational experience, especially since it was my first time participating in a Capture the Flag format. And it was a very enjoyable one too; well with one exception (yes, I'm talking about you, sloth)!

I would like to present here a summary of my approaches to all the challenges along with their corresponding notebooks where it's necessary.  I hope it helps.

- **Math**: I just used DBSCAN and PCA to find the clusters and the dimensions for each subtask [[Code](https://www.kaggle.com/code/vkonstantakos/math-challenges)].
- **Hotdog and Hotterdog**: Overlayed a hotdog picture over Chester for the first one, and used a general gradient attack to fool the network for the second one (by using InceptionV3 and MobileNet as proxies) [[Code](https://www.kaggle.com/code/vkonstantakos/hotdog-challenges)].
- **Bad to good**: I did not find a good automatic solution here. I tried different values for various students, guided by DBSCAN clustering results [[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)].
- **Baseball**: Tried different distributions with various bounds to understand the pitches of every player. Then tried to imitate the pitches of Henry manually [[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)].
- **DeepFake**: Used a pretrained First Order Motion Model to generate a quick DeepFake video [[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)].
- **Honor Student**: Simply edited the F to look like an A [[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)].
- **Theft and Salt**: Used the [ART ](https://github.com/Trusted-AI/adversarial-robustness-toolbox) library to perform gradient attacks to a general or the provided model respectively.
- **Token**: Checked for possible keywords that can 'fool' (desync) the tokenizer (e.g., SECRET, BLANK, and corresponding word stems) [[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)].
- **WAF**: Tried various exploits from [here](https://github.com/payloadbox/command-injection-payload-list) until I found the one; then used space to bypass it [[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)].
- **Inference**: Used Kaggle handwritten characters datasets to try various letters until I got a very manageable search space. Then tried the combinations in a loop [[Code](https://www.kaggle.com/code/vkonstantakos/inference-challenge)].
- **Forensics**: Just the model summary [[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)].
- **Leakage**: Used the provided model to generate character-by-character predictions starting with the username [[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)].
- **Murderbot**: Trained multiple classifiers and took their weighted average as the final prediction [[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)].
- **Wi-Fi**: Visual solution: extracted the tokens, plotted them in different embedding spaces, followed the path, and got the flag [[Code](https://www.kaggle.com/code/vkonstantakos/wi-fi-challenge)].
- **Crop1:** Used Bayesian Optimization with Optuna to minimize the scoring function. Genetic algorithms also did the job.
- **Crop2**: ?? (Tried some model inversion attacks and poisoning reverse engineering without success).
- **Sloth**: I don't want to talk about this...([[Code](https://www.kaggle.com/code/vkonstantakos/miscellaneous-challenges)], [[Meme](https://www.kaggle.com/competitions/ai-village-ctf/discussion/351803)]).

More notebooks will come as soon as I clean up the code from all the different sources.

Again, thank you all here. I really enjoyed this competition.
