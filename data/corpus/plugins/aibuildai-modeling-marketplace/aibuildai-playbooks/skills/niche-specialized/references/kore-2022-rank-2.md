# 2nd place solution - Rule based

Competition: kore-2022
Rank: #2
Source: https://www.kaggle.com/c/kore-2022/discussion/340994

## Introduction

My solution, like many others, is based on the baseline solution provided by @egrehbbt. I'll try to describe the changes I've made to it and hopefully, I won't miss anything important. If you have any questions, feel free to ask them.

## Planning

First important change I made was that I added planning to all the Actions. If I wanted to create a new shipyard, but didn't have enough ships, I would check whether I would have enough in the next 5 moves. If the answer was yes, I would either spawn ships or do nothing. Same was true for fleet and shipyard attacks. For mining, planning was a bit different. I would compare my current best route's score(expected kore divided by route's length) to the future ones. If the future ones were sufficiently better, I would wait(i.e. spawn or do nothing).

## Main shipyard

It turns out that you often don't need a lot of ships to mine effectively. This is because increase in % mined is logarithmic with regards to the fleet's size. In fact, many competitors noticed the opposite problem - mining too well and quickly depleting kore near their shipyards. So, what to do with all those ships that are not needed for mining? You can either create new shipyards or use them for offense. I decided to do the latter(which wasn't always the right choice). I chose one shipyard to be my main shipyard, based on distance to the closest enemy. If I had more shipyards that were equally close, the one with most ships(taking into account incoming fleets) would be the main one. Then I would send ships to that shipyard until it had a certain % of my total number of ships. Percentage ranged from 55 to 85, depending on how close the closest enemy shipyard was. I also had some other rules for distributing ships between shipyards, but those aren't as important so I won't describe them here. Having 70%+ of my ships at a single shipyard often allowed me to take over enemy's shipyards with ease, specially if they were close to me.

## Converting leads into victories 

### big_attack

Sometimes I would lose games even though I had a large advantage(though this was more common when playing against my own agents). To combat this, I created big_attack - a macro level strategy, which involved multiple shipyards and would last 10-15 steps. If a condition was met, I would choose 1 shipyard to lead the attack. Other shipyards would spawn for as long as possible before sending ships to the chosen one. Then the chosen shipyard would attack. I considered doing coordinated attacks like others(i.e. attacking from multiple shipyards at the same time instead of aggregating ships at 1 shipyard first), but decided against it because I didn't think it would make a big difference.

### crash_ships

Another way of dealing with this was crash_ships, which is similar to "whittle attacks" used by @itswin . If I had lots of ships at some shipyard(usually my main shipyard), I would attack an enemy's shipyard even though I couldn't capture it, with the intent of reducing that shipyard's ship count and causing other difficulties.

Some of my agents had crash_ships or big_attack turned off(I always used at least 1 of them). Agents with only big_attack did slightly better than the agents with only crash_ships, and the best agent had both turned on.

## Expansion

Expansion is one of the most important parts of Kore 2022. Even seemingly small changes in the expansion strategy would produce drastically different outcomes. There are 2 main questions here: **where** and **when**.

### Where

Instead of just looking at the target point's nearby kore, I weighted the values based on distance. If a point was closer to the target point, its kore would be multiplied by a higher number. Other than that, I tried to expand not too close to the enemy and not too close to my other shipyards. I had 3 separate rules(for the first expansion, second, all the others), but the differences aren't significant enough to write about.

### When

As time went on, I gravitated more and more towards faster, less safe expansions. It wasn't obvious whether this was good(specially because my agents with reckless expansions were doing worse initially), but I saw that agents in the top 5 weren't as aggressive with their early attacks, and so I thought I could get away with it. In the end, I was right. Even though it took them longer to converge, agents with unsafe expansions did noticeably better. My first expansion was the least safe, while for others I calculated safety based on the number of ships, available kore and distance to the closest enemy.

Other notable changes I made to the original solution are:

- any shipyard could expand at any point(while in the original only the shipyard closest to the point could expand at it)

- I would never create a new shipyard if the number of my shipyards was greater than the enemy's by 2 or more(including ongoing expansions).

## Defense

Improving defense was very important in order to ensure that I could survive at least some of the early attacks. 3 major changes I made to the original solution are:

- Mining with guard ships during the attack(the only difference between guard ships and regular ships was that guard ships had to be back in time to defend, so their routes were usually much shorter)

- Sending defense even when it's not on time(up to 8 steps late). Sometimes I wasn't able to defend a shipyard, but that didn't mean I should give up on it. This sometimes allowed me to recapture a lost shipyard.

- Sending defense early. In the original solution, no matter how many ships you had (at other shipyards), shipyards would wait until the last moment, when they would send ships to the attacked shipyard. But sometimes they would have enough ships long before that, and it was ineffective to wait so long(attacks like crash_ships could exploit this weakness), so I would send defense as soon as I had enough ships.

## Fleet attacks, counter-attacks and mining

Not much to say about this. Besides planning which I mentioned earlier, I did mostly the same as others(added more routes, checked safety, direct/adjacent attacks, attacked fleets that were attacking my fleets etc.), but I think this is the weakest part of my solution and the one that could be improved the most. In many games, my fleets were getting attacked a lot and I would lose large amounts of kore because of it. Making mining safer gave me worse results both locally and against others. This means that my way of measuring safety wasn't good enough and I needed to take more details into account.

## Preventing first expansion

In the original solution's expansion code, there's a rule:

```
  if incoming_hostile_fleets:
      continue
```

This is generally a good idea, but it can be exploited - by sending very small attacks often. I
added this to my final submissions and to my surprise, it worked. Well, sort of. It increased my rating by ~150 points(my best rated agent without this attack is 1755), but it didn't change my position on the leaderboard.

## Closing thoughts

While I enjoyed the competition and the psychological aspects of it, I think it's unfortunate that so many solutions were rule based. I'm looking forward to the future competitions and hope to see more RL there.
