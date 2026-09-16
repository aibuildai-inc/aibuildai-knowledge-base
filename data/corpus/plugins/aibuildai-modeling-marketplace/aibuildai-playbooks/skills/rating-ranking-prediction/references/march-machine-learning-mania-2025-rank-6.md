# 6th Place Solution: An Attempt to Bet Responsibly

Competition: march-machine-learning-mania-2025
Rank: #6
Source: https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572482

For this year’s March Madness competition, I wanted to opt for a high-risk, high-reward approach. First, I needed a good baseline—preferably sticking close to bookmaker odds—since the bookmakers/Vegas have both the means and the incentive to set accurate odds. Luckily, [@kaito510 provided an excellent baseline here](https://www.kaggle.com/code/kaito510/updated-goto-conversion-winning-solution), to the point that this part could be concluded without even needing to do any ensembling. I used the version without any overrides, because by default the notebook already includes a deviation from bookmaker odds in the form of an override on the Drake–Missouri game.

The second part was to add a bet on top of this baseline. Adding the bet can be approached in different ways—some better than others—depending on what your objective is (e.g., bronze medal, silver medal, gold medal, etc.). I set my goal to optimizing for top 8 probability. To approach this, I thought the best way was to go for a **concentrated, decorrelated bet**.

By “concentrated,” I mean deviating from the baseline on only a few games. Concentration is great for maximizing the probability of hitting a certain Brier loss threshold, with the optimal number of games in this case being one (neglecting some loss of flexibility in risk tolerance). However, for this competition one does not want to simply optimize for a Brier loss threshold, but instead for a **rank threshold**. Though the two are correlated, optimizing for Brier loss ignores the **adversarial** aspect of the competition.

This is where **decorrelation** comes in. I did not want the Drake–Missouri override because, if that upset happened, I would likely be competing with a lot of similar (correlated) solutions. One should try to bet differently from others, so that if one gets lucky, the chance of success is not jeopardized by others who get lucky in a correlated way. In other words, it is good to position in a sparse region of the outcome space.

Decorrelating from other concentrated bets is pretty straightforward under the assumption that you know which ones may be popular. But I think it is also useful to think about **non-concentrated** bets—i.e., **diluted bets**. Many people, knowingly or unknowingly, employ diluted bets when using bookmaker odds as the reference. To decorrelate from diluted bets, one has to expand their bet into more than one dimension, at the cost of concentration, since the magnitude of the cosine similarity is always 1 between one-dimensional (non-zero) vectors.

I conjecture that most diluted bets either boost the favorites or boost the underdogs, which is convenient, since this allows for adequate decorrelation in just two dimensions. To decorrelate, one should then override one favorite and one underdog to win their respective games.

The final step is deciding how much risk to take on to maximize the probability of hitting the desired rank threshold. Here, I was left to just guessing, as I do not have a way to compute this. I think my participation in the 2023 edition may have helped shape my intuition a bit.

With a loose estimate of the required risk, I opted to bet on two round one games: **St. Mary’s (CA)** winning over **Vanderbilt**, and **McNeese St.** winning over **Clemson**.

**Final note:** If I had not undone the Drake bet in the goto-conversion notebook, I would have won the competition. But I have no regrets, since including this bet would have exceeded my personal risk tolerance.
