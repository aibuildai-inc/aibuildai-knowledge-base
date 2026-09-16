# 12th (ish) placed agent. Top rules-based agent?

Competition: lux-ai-2021
Rank: #15
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/293953

[https://i.imgur.com/uUPUK9K.png]

# The strategy:

Take tree clusters. Use carts to ship resources to those clusters, then use chain transfers to extract wood that's safely inside for late game city-tiles.

I went for this strategy as I decided holding clusters was a hard problem to solve for a rules based agent. 1v1 blocking is fine. 2v2 blocking is harder but doable. XvY blocking with potential city tile unit builds wasn't something I wanted to tackle. While I just said 1v1 and 2v2 blocking were ok, even they have their challenges when you're trying to decide if you actually want to block or whether something else is more important. For example: in 1v1 maybe you'd rather "attack" an opponent cluster rather than defend your own. There's even more to it than that, suffice it to say that behaviour that looks quite simple has a lot of opportunity to go wrong. 

Building city tiles around the clusters seemed to reduce the chance that I'd have to make hard decisions about blocking. The trees are safe, and we can work from there. Turns out this wasn't the best strategy. Everyone else in the top 10 is able to hold their clusters and if you're able to do that, it turns out that building nearer to the coal/uranium and having a impenetrable blocking algorithm is stronger. This makes sense - it means you don't have to do complicated things to extract the wood, control more "land" and have much more certainty about expansion as you don't have to account for resources that'll soon arrive at your major cities - the major cities are at the resources so they arrive nearly instantly.

I can see why ML agents beat rules based agents, with hindsight. There are just so many cases to consider. Even the first move is really not clear cut and my agent often got it wrong in a way that was fairly obvious for me to see, but hard to code. There's a lot of trade-offs to be made even here. Do you go for building a city fast? Do you push toward the opponent? Do you push toward clusters of resources? What if the cluster is closer? Further? Larger? Smaller? There's a lot of factors to take into account here and it's rare that you're building a city fast that pushes you toward the opponent and toward fresh clusters. This is something RL can balance quite well, but is a real pain for a human. It's very possible to lose a match by making the wrong first move.

So, all in all, I did the wrong thing. Rules were the wrong choice. The strategy was wrong. The implementation could still use more work. But it kinda did ok. The gold medal is still pretty close to being lost. Only 3 points and two places in it at the time of writing. I reckon my current odds at holding it around around 60%.


# The implementation:

I mostly went for a stateless approach. I iterate across all the units in multiple-passes scoring the actions that the units may take with the context of what other units/cities are doing. If there's a clash, units are rerun with knowledge of what caused the clash. Standard A* pathfinding with weights to bias toward/away from certain things. I also built tables of path length approximations avoiding cities using Dijkstra which were for fast lookup to avoid having to run the full pathfinding too frequently. 

There's a few places with state, mostly to hack around unstable scoring. For example, there is specific code designed to reach new clusters. Scoring where exactly to move to came out fairly unstable, so there's a bias toward doing whatever it is you planned to do last tick. I still get a bit of "dancing" though where performing an action results in the weights changing and the unit wanting to go the opposite direction next tick. Maybe more statefulness would stabilize this.

In general, I don't think there's anything too surprising in my code. I tried at one point to try and tie RL in there as additional biases to my hand-crafted heuristics, but nothing I tried worked. I would have liked to continue down this avenue as maybe it would have resulted in something quite interesting, but as I was doing this I started to drop down the leaderboards and decided it'd be safer to improve what I already had.

# Stuff I learned

1. Self-play is often a bad proxy for real performance. I had a lot of self-play episodes that came out like this:

	[https://i.imgur.com/pn8LmZG.png]

	This is because small iterations of my agent tended to cancel each other out. My "attacks" we well countered by my "defence" (as that was easiest to test). Similarly mistakes in one agent often exist in the other, so are not exploited. As such both agents got a lot of resource, and could exploit those captured trees in the end game. This is not how many of my matches on the leaderboard go. 

	I spent a reasonable amount of time downloading replays, identifying mistakes against other players, and fix them. In general I needed to do more of this earlier, and less relying on self-play. I did have a "mutator" that would change my agent's parameters around with the aim of making it diverse and presenting more interesting replays, but that didn't help much.

2. Similar to the above, self play takes time. I often had two agents where there was a clear winner after 100 matches, which was no longer clear at 200 matches. Especially on larger maps my agent could easily eat 0.5-1s/turn on avarage, so 100 matches was not quick.

3. Lux had a lot more time per tick than I was used to. I think early on in development I spent a bit too much time trying to find neat solutions to something that could be brute-forced in under 10ms/tick.


# Feedback on the competition:

- The fact that so many imitation agents made it to gold/the prizes is a weakness of the competition setup IMO. While I appreciate the challenge that comes with creating an imitation agent is not insignificant, and I have nothing against the people who went down this route (a competition is a competition!), I don't think it creates a particularly interesting environment to compete in. Having one or two strong agents and most of the rest of the field being "who can copy them the best" seems rather uninspiring. I don't know for sure, but I suspect 7-8 of the gold medal/prize winners will be just imitating better agents. I don't have any suggestions to avoid this within the Kaggle format. 

- I think the competition should have stuck closer to it's theme. Building cities with coal or uranium doesn't sit quite right with me. Nor does intentionally letting city tiles die. I've given that feedback before though and the devs disagreed. 

- It feels like the devs failed in their goal to allow for a number of different methods and strategies to be successful. As per the title, I suspect my agent may be the top rules based agent, and it's easily 400 points off the top RL based agent. The top two RL based agents both use fundamentally the same strategy, and all the agents imitating them are again the same. It feels like a single strategy is significantly better (though with only 2 or 3 near the top actually innovating on strategy, it's hard to be sure), and a single implementation method is significantly better. Building a balanced game without being able to change the rules as you go is very hard and I don't think we could have predicted this going into the competition, but that doesn't change the fact that they failed to get the diversity they were hoping for. Wishing them the best of luck for Lux Season 2. 

# Closing

I'm on the fence about participating in Lux Season 2. On one hand, I do enjoy a good coding strategy game. On the other hand, I'm feeling pretty jaded about the power of machine learning (and particularly imitation) to outperform methods I consider to be fun. I get that other people find machine learning to be fun, but that's not me! I think if I found that someone 100+ points ahead of me was also using rules I'd be more satisfied, as it'd mean I would know I could do better, but as it is I don't feel like the aspect of coding strategy games I enjoy is going to be a competitive solution in competitions that don't make it significantly harder for machine learning methods to operate.
