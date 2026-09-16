# 15th place solution - Don't Test At Home

Competition: rock-paper-scissors
Rank: #15
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221696

Early on in the competition I was heavily influenced by [This Notebook](https://www.kaggle.com/ihelon/rock-paper-scissors-agents-comparison) and at the time how effective the decision tree agent was. The next logical step was to do an ensemble of decision trees. There are a lot of flavors of this sort of ensemble, but I decided to go down the 'Extra Trees' route. It quickly became apparent that the number of trees that could be included in the ensemble seemed to be dependent on the server load/number of steps being fit with some steps only allowing fitting of 100 trees and other steps 800+. So to address this I began using an internal timer in my agent to count how many milliseconds had passed since my agent was given control. My agent would then fit as many trees (random splitter and max feature of 1) as it could in the first 950 milliseconds and then accumulate the probabilities from each tree and use them to weight np.random.choice, and that's it a pretty simple agent that got 15th place and might have won the first place medal for largest computational cost of all agents.

Thanks for a great competition!
Best,
Dominique
