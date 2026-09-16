# 6th place solution つ ◕_◕ ༽つTAKE MY ENERGY ༼ つ ◕_◕ ༽つ

Competition: rock-paper-scissors
Rank: #6
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221494

My full code: https://www.kaggle.com/returnofsputnik/votesprob-ensv2-40-sm4-nwin-nlos-gain-loss

### Introduction

From November 2020 to February 2021, an international Rock Paper Scissors competition was held on the website Kaggle. It is such a simple game and finally one that I could compete without borrowing hardware from the cloud. I am happy to say I achieved a solo gold medal coming in 6th place (in fact I had 7 submissions that would have given me a gold medal).

### The Competition Game Rules

In this competition, you had to create a bot that throws Rock, Paper, or Scissors to defeat the opponent. The game lasts 1000 rounds. You get 1 point for winning a round, -1 points for losing a round, and 0 ponits for a draw. After 1000 rounds, if your score is at least +20 then it is marked as a "won match"; if your score is -20 or less then it's a "lost match", and any scores between -20 and +20 are considered a "tied match".

Each bot has a ranking assigned to it based on how much it wins and loses. For example, if you defeat an opponent, your ranking goes up and their ranking goes down. If you tie but your opponent was better than you, then also your ranking goes up slightly and their ranking goes down slightly. Similarly, if you tie but you have a better rank, your ranking goes down slightly while theirs goes up slightly.

Bots mostly played other bots nearby their ranking but also 33% of their matches were randomly paired.

### Strategy

At first glance, the most obvious strategy is just to play random every single time, that way your opponent can never guess you. The problem with that is that some opponents are not random and thus predictable. For example, if your opponent played Rock on every round, then a totally random bot would only win 50% of the time, whereas a non-random smart bot could win 100% of the time. When some players are known to be sub-optimal, it is absolutely essential to try to detect patterns and tendencies in the play of the opponent, and then employ an appropriate counter-strategy. (Source: https://webdocs.cs.ualberta.ca/~darse/rsb-results1.html)

Therefore a non-random bot is best in this competition. I needed to make a bot that could predict and defeat the opponent's next move reliably. Furthermore, I needed to make sure I won by at least 20 points to be considered a "won match" and make sure I minimized the number of games I lost by -20 points.

Designing a smart bot is not straightforward due to all the headaches of "outsmarting strategies". Luckily there have been previous Rock Paper Scissors competitions from which I could re-use strategies and code.

By reading the results of the 1999 RoShamBo competition held by The University of Alberta, lots of smart strategies that I used can be found:
1. Create a "bailout strategy" - i.e. if you are getting close to losing by -20 points, then switch to a random strategy to try to lock in a draw.
2. Have a history decay function to favor more recent actions by your opponent, which combats drifting opponent bots.
3. Look for the presence of a pattern versus the absence of a pattern

### Progress toward my solution

Using code from the previous Rock Paper Scissors competition, I submitted "rps_meta_fix.py" and it did pretty well on the leaderboard. However one day my bot played the #1 on the leaderboard, Stas SI, and it lost by over 100 points! And that same day I studied Stas SI's matches and I noticed many games he was winning by over 950 points!

This insight made me realize that I could create anti-bots that perfectly defeat popular deterministic bots like "rps_meta_fix.py" and "dllu.py". Most of the public bots in the competition were totally deterministic, which means that if you know their algorithm, you can perfectly predict their next output. Basically, you pretend and use your output as their input, then defeat what you expect them to output.

So now my strategy was to have an ensemble of public code bots along with "anti-bot" versions of them. Whichever bot is performing the best, I select that bot to throw the next hand. This is the crux of my strategy. Already by my 24th submission I would have finished in 10th place on the LB.

In order to not be discovered like how I discovered Stas SI, I put in my code that if I were winning by >= 60 points, then I would switch to a random strategy so my opponents would not get too suspicious of me.

There was also a period of a few weeks where I did not submit for fear of other people downloading my games and learning my algorithm. However I later realized that due to the randomness of Rock Paper Scissors, it was better for me to submit as many times as possible to maximize my chances.

### Solution

My solution is an ensemble of 50 bots and 50 anti-bots, yielding 100 total bots. All of these bots are publicly available online. I grade each bot on their performance over the last 40 rounds. If they were winning over the last few rounds, their hand got added to the list weighted by its score. Then I randomly choose the hand from the distribution of advised outputs. Crucially, I change the hand probabilities to expected gains, so instead of Pr(Win) I use Pr(Win) - Pr(Lose). If I'm winning by >= 60 points, then switch to random. The entire submission is around 7000 lines of code due to all of the bot algorithms.

The contribution I made was the "anti-bot" insight and also understanding each public bot enough to actually write the anti-bot.

Towards the competition end I had created a variant that instead of each bot returning its proposed hand, the bot returned a probability of each Rock Paper Scissors. I believe this way is superior, but it got 9th place instead of 6th place (still fantastic!)

Unlike other opponents, I did not submit the same bot multiple times to the Leaderboard to get its rank distribution. Instead, I submitted variants of the same bot, hoping some would perform well.

I experimented with adding improvements:
1. Throwing the first N hands as random - this way my opponents relying on history matching would be confused
2. Applying a meta rotation across my outputs in case my opponents were tricky
3. Switching to a 50% random strategy if my score is <= -5
4. Grading my bot ensemble using the last 40, 50, 60, ..., 100 moves
5. Decaying my bot wins over the last 40, 50, 60, ..., 100 moves
6. Choosing the top 30% of my bots and randomly picking a move from them, weighted by how good they have been winning
7. Rather than choosing the hand that wins, choosing the hand that doesn't lose (e.g. if I predict Paper, then rather than choosing Scissors it randomly picks between Scissors and Paper, i.e. win or tie)
8. If my chosen bot loses, "drop switch" it to a different one
9. If my chosen bot loses, implement a cooldown period where it can't be chosen for another 5 moves
10. Rather than looking at the best bot over wins minus losses, pick based on their sharpe ratio, or based on dirichlet weighting, or play meta and pick whichever is performing the best long-term

Later on I experimented with copying Stas SI's games since he was so good on the leaderboard. I expanded this to other strong gold positions on the leaderboard. The idea was that if Stas SI had seen the exact previous 8 hands or so, then play his exact response. These bots do okay but not great since there isn't enough data to make it strong enough.

I also knew that using Reinforcement Learning would not be a good approach. By reading about Facebook's Reinforcement Learning algorithm (https://ai.facebook.com/blog/rebel-a-general-game-playing-ai-bot-that-excels-at-poker-and-more/) I learned that RL bots try to converge to the Nash Equilibrium of their game. The problem with this is that the Nash Equilibrium for Rock Paper Scissors is to throw random every round, and we already know why this is not the best approach. Therefore I did not experiment with RL or models in general.

### Shoutouts

Huge congrats to "Boooooooow", Georg Streich, Stas SI, Taaha Khan, and others for such strong finishes and also for sharing on the Discussion forums. Congrats and thank you also to some of the Discussion contributors - Charles Eric, John Termaat, glazed, Chan Kha Vu, Nick John, Tony Robinson, James McGuigan, Daniel Lu, Ant, Robga and others (let me know if I forgot you!). Sorry for those who came just outside of gold.

I am moving toward Kaggle Snake competition. I already have seen some familiar faces. Cheers!
