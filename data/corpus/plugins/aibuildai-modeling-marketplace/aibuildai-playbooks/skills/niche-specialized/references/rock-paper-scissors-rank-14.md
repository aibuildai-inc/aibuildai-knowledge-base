# 14th Place Solution

Competition: rock-paper-scissors
Rank: #14
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221590

This is my first time to write a solution. Any questions or suggestions would be welcome.

My best agent is a fixed version of Ilia's Multi-armed bandit agent. 
About 20 strong agents from RPS contest and anti-agents are included in agents pool.  Simple agents like mirror or popular beater are also included. 
I tried many combinations of step size and decay rate of MAB. In case of my best agent, step size is 4 and decay rate is 1/1.05. Beta distribution is used to decide which agent is most probable to win.


**Experiments**
In the first week, I made some rule-based agents (mainly transition matrix). I read some articles related to RPS and I found RPS contest website. Agents from RPS contest worked very well then (gold - silver). 
When these agents were no longer winning, I started combination of these agents and MAB. Many combinations of step size (1 - 4) and decay rate (1.00 - 1.50) were tried. 

Next step was introduction of Dirichlet distribution. Beta distribution has 2 parameters but RPS has 3 states (win, loss, tie). Dirichlet distribution was considered to be more appropriate to RPS. Dirichlet agents worked better than Beta agents on average. Step size and decay rate were chosen manually as in Beta agents.

Best step size and decay rate could be different between first 100 rounds and last. I adopted another Dirichlet model which calculated winning rate for each Dirichlet agent and decided step size and decay rate in every round. Even though parameters of this model were needed to chosen manually, parameters of Dirichlet agents to choose which agent was likely to win had come to be chosen automatically.

**Result**
As a result, this type agents couldn't keep gold medal position. In the last few week, Beta agent surprisingly worked well and finished this competition in 14th place.

Thank you for reading here. I'm not a native English speaker so please forgive me if my English sounds strange.
