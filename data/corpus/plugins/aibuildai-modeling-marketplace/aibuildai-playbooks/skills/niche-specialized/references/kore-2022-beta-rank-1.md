# 1st Place Solution

Competition: kore-2022-beta
Rank: #1
Source: https://www.kaggle.com/c/kore-2022-beta/discussion/317737

Ahoy, there!

My solution is a simple rules-based agent which sequentially performs some basic operations:

- shipyard defence - We can find out where and when our shipyard will be attacked, and we have time to form a defense. Knowing how many ships are currently in the shipyard and how many my ships will arrive in the near future, I can say exactly what minimum number of additional ships I need to get for a successful defense. If the shipyard has enough spawning capacity and time, then I just create the necessary number of ships in the shipyard. But if that's not enough, then I will send ships from the nearest shipyards to help.

- shipyard attack - The same thing, I know the exact number of ships that need to be sent to capture a shipyard. Unfortunately it doesn't always work.

- direct attack - I launch a pirate fleet to intercept an enemy fleet, just as simple as that. And of course I choose a route that will not overlap with the routes of enemy fleets.

- adjacent attack- It's looks like this: . I sacrifice my fleet, but at the same time I create double or even triple damage to the opponent.

- expansion - I create new shipyards if I have an surplus of kore, and my current spawning capacity is much less than I can mine.

-  mining - I look at the huge number of possible routes that are available to me, and choose the one that will give me the largest number of kores per turn. At the same time, I check that a route does not intersect with routes of opponent's fleets. This was very important in 4p games, but I'm not sure how much it's necessary in 2p games.

- spawn - I spawn as much as possible until my fleet outnumbers the opponent's fleet by several times.

I can share the code, but I'm not sure if it would be appropriate. Will it harm the upcoming competition? What do you think?

I hope you enjoyed this little competition as much as I did. See you in the next one!

---

Edit: you can find my code [here](https://github.com/w9PcJLyb/kore-beta-bot)
