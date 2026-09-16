# A solution that I hope will earn a silver medal (currently in 29th place)

Competition: lux-ai-season-3
Rank: #25
Source: https://www.kaggle.com/c/lux-ai-season-3/discussion/567581

We would like to thank everyone who participated in the LuxAI Competition for all of your hard work.

We truly enjoyed the competition, though we did lose a little sleep. (>_<)

Although we didn't quite reach the gold medal position, we were able to finish with a silver medal as of now.

---

### Foundation — The Exceptional Notebook

Our approach is built upon the exceptional notebook published by @egrehbbt, which served as the core of our strategy:
https://www.kaggle.com/code/egrehbbt/relicbound-bot-for-lux-ai-s3-competition

We would like to express our deep gratitude to @egrehbbt for providing such an outstanding resource. Building on this foundation, we developed a simple, rules-based solution with five main modifications.



### Modification 1: find_relics intensive mode

When a relic has not yet been found but either our points or our opponent’s points are increasing (i.e., a reward is appearing), we redirect all ships to focus on searching for relics.

In this game, the most important factor is ensuring we do not fall behind our opponent in finding relics.


### Modification 2: Harvesting – Energy Charge – SAP

In addition to harvesting, ships recharge their energy for a later “avalanche.”

We also use SAPs to intercept or hinder ships moving into our area.


### Modification 3: “Avalanche”

Toward the end of the game, ships that have accumulated energy are grouped together to attack the opposing side’s harvest.

By launching a synchronized assault, we can overwhelm our opponents at a high rate. The key is to turn the game around in the last 10 steps.


### Modification 4: Predicting board changes

By observing the timing of board changes and energy accumulation, we can accurately anticipate future states.

We then chart the optimal route while considering these impending board changes.


### Modification 5: Blind SAP

We can deploy SAP even when opponents are not in sight.

By estimating how much of the opponent’s reward is filled (based on how many points they have gained), we use SAP to disrupt their progress even without directly seeing them.

---

In this competition, I teamed up with my colleague, Kazuhiro.O. Early on, our scores were similar, but as the competition progressed, his score climbed significantly. Meanwhile, I struggled to increase my own, so I helped him by optimizing his program’s speed.

Kazuhiro.O is an excellent colleague who holds a Grandmaster rank on SIGNATE (a Japanese platform similar to Kaggle). I’m genuinely grateful that we could collaborate for this competition.
