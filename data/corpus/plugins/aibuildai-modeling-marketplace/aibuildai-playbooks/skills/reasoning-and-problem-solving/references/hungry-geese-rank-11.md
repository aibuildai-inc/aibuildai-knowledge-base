# 11th place solution(NN+MCTS)

Competition: hungry-geese
Rank: #11
Source: https://www.kaggle.com/c/hungry-geese/discussion/263655

Congrats to the winners. Thanks to team members ( @iiyamaiiyama @kibuna ) and Kaggle team for hosting a great competition.
Our solution is a combination of NN and search strategy(MCTS).  I will share our solution in detail. 

Summary
* Model + MCTS

## Model
We used 3 stage training.

1. Imitation learning using above 1230.
2. Finetune HandyRL(self training)
3. Re-train Imitation learning using above 1230.

### Training Parameters
* Model Arch: ResNeSt
* Optimizer: SAM(Adam)
* Feature
  * Head/Tail/Body each player.
  * t-1,t+1,t+2 tail
  * Food position
  * distance of food
* Augmentation
  * Player Shuffle Augmentation
     * 1-3 is shuffled, 0 is own player, it's can't be shuffle.
  * H/V Flip

### Inference Speed
Torch Script compile trained model for inference speed up but model quantization is slower than original.

## MCTS
  Monte Carlo Tree Search with Decoupled UCT(UCB1)

* UCB1  
We used a simple Decoupled UCT with UCB1. In typical MCTS, playout is performed until end of the game, but we limited it up to 10 turns. This is because playout until end of the game is very time-consuming and the random factor is too large to evaluate properly.  
We do not run CNN during MCTS like Alpha-zero. It takes too much time too.

* Reward  
If the game is over, it is easy to calculate a reward used by ucb. However, as mentioned above, many playouts are terminated in the middle of the game. Therefore, we decided to give the rewards as follows:  
  * if game is over: give them score according to final rank. (e.g. 1st place=1.0, ..., 4th place=0.0)
  * if game is not over: 
    * if agent is dead: give 0
    * else:
      * if other one or two agents are dead: give 0.7
      * else: give 0.5
  * misc rewards:   
    Taking food: +0.01  
    Chasing its own tail: + 0.002(per turn)
    
* Crazy-Move during playout  
The quality of playout is very important factor in MCTS, and random movement in the playout will not give proper results.  
Of course, as the number of playouts increases, it is expected that UCB-value will gradually converge to the optimal movement. But terminal nodes of the game tree cannot have enough playout opportunity. Therefore, when the MCTS visits a board that it has never seen before, it selects moves based on [crazy-goose agent](https://www.kaggle.com/gabrielmilan/crazy-goose).  
When the board is visited for the second time or later, MCTS selects moves based on the UCB1.  

* Anti-public agent  
There are many public kernel clone(copy) agents in LB. As long as it works deterministically, it is easy to find out that it is a clone or not. If we know that it is a clone agent, we can predict the agent's next move 100% and take advantage of it.  
However, it is impossible to infer the clone agent's move for every board visited during playout due to 1 second limitation.  
Therefore, we predict the clone agent's movement only in the first board (root node) where the MCTS starts.
Although there are few clone agents in silver-gold zone, exploiting them stabilizes the win rate in the first 10 games after submitting. This speeds up the convergence of LB and makes it easier to select a stronger agent for us.  
In addition, we expect that in the final evaluation week, we match clone agents who happen to win a lot of games due to the increased sigma. However, the sigma increase in this competition was very slight, so the contribution in this aspect was small.

* C++ implementation  
  We implement MCTS in C++ and call it from python.  
  * Number of playout:  
    Python implementation: 600  
    C++ implementation : 4000  

## Ensemble
argmax(model probability * MCTS score)

## Things that didn't work
* AlphaZero-like tree search
* UCB1-tuned
* Food respawn during playout
