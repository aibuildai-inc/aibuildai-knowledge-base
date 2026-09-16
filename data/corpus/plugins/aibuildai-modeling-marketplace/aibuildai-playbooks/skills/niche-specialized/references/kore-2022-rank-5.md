# [1 Musketeer] 5th place solution: Rules based

Competition: kore-2022
Rank: #5
Source: https://www.kaggle.com/c/kore-2022/discussion/339979

## Preface
First of all, I want to thank the Kaggle team for hosting a lovely competition. I greatly enjoyed seeing the strategies develop over time and reading about people's thoughts about the game on the forums. I wish I had more time to work on my agent, as there was so much more to be done, but I look forward to future simulation competitions on Kaggle! Here are also a few people I'd like to thank in particular:
- @egrehbbt for posting his amazing [code](https://www.kaggle.com/competitions/kore-2022-beta/discussion/317737) from the beta competition. I used it as a jumping off point and it saved me more time than I could ever imagine.
- @jmerle for his AMAZING visualization tool [Koreye](https://jmerle.github.io/koreye-2022/).
- @bovard for posting the competition on the Battlecode discord, as I probably wouldn't have found it otherwise.
- @kirderf and @jaroslawbogusz for their local evaluation and replay scripts respectively.

I'm a Battlecoder by heart and new to Kaggle, so I took a rules-based approach to this competition. If you don't know what [Battlecode](https://battlecode.org/) is, definitely check out Stone Tao's survey of AI programming challenges [here](https://www.stoneztao.com/blog/posts/ai-challenge-survey/). But, in short, it's another AI real-time-strategy game that I treat like a full time job in the (seemingly short) month that it runs.

## Overview
At a high level, I followed the same basic decision sequence layed out in aDg4b's Beta solution. Here's an overview of my bot and my thoughts throughout the competition. Where I can, I'll try to give some reasoning behind some of the decisions I made, or talk about things that I tried but didn't end up working out for me.

### Defence
See enemy ships? Send reinforcements. That's defence in a nutshell. At a deeper level, my bot checks all of its incoming allied and hostile fleets and calculates the minimum accumulated difference at each time step. If it's negative, it spawns and requests help from other shipyards if either (1) those shipyards can help by the time the closest enemy fleet gets there or (2) I only have a few shipyards and the one under attack is one I built myself.

On the other hand, sometimes you simply can't send or generate help quick enough. Maybe your opponent coordinated their ships better than you. Maybe they simply had better efficiency in the early game and out-mined you. In a case where my shipyard realizes it can't defend itself anymore, it defers to any other launching action and sends a hail mary fleet. Most of the time this just served the purpose of sending a (likely large) fleet to reduce the kore around the area, but it had another interesting side effect too. When I've just captured an enemy's shipyard, if they send all of their ships at once to defend and leave another shipyard helpless, my shipyard will launch a fleet to capture it rather than doing nothing and losing the ships. This results in killing their ability to spawn by resetting both (or more!) enemy shipyard's turns controlled. Since Kore is as much a game about getting resources as using them, this normally wins the game on the spot.

**A failure:** I experimented with sending exactly the number of ships needed for help, but this often backfires as a shipyard that's getting attacked now is also probably going to be attacked in the future. Here I was basically taking the philosophy of overcorrecting now and worrying about other issues later. As long as I didn't end up in the situation above, i.e. sending all of my ships at once, this defence generally performed better.

### Offence
My offence related to attacking shipyards consisted of three different types of attacks: captures, coordinated captures, and whittle attacks. In general I was super conservative with offence, only trying to capture when I was **sure** it would be successful. There were some times where this wouldn't be perfect, as I wouldn't be able to predict new mining fleets an enemy would send out, but in most cases taking into account an enemy's kore/cargo for spawning and the reinforcements from nearby shipyards was enough.

- Coordinated captures took first precedence. This attack consisted of syncing up attack fleets from multiple shipyards to focus on a single shipyard. By not launching them all at the same time, I thought it would make it a little harder to coordinate defence.

- Normal captures were the same, except using a fleet from a single shipyard. Shipyards would consider waiting for an attack if it predicted that it could take the shipyard even considering the extra reinforcements the enemy would get in that time. Note that this also had the effect of me being able to choose the maximum kore route to attack the shipyard, since I knew with almost complete certainty that I would convert the shipyard and get the kore.

- Whittle attacks were the least common, and generally served to extend an already established lead. Upon gaining a lead in ships, these attacks consisted of sending fleets of size 50 or more to an enemy shipyard, with the idea being that a difference in ships of 120 to 60 is much harder for the opponent to come back from than a difference of 170 to 110. This also had the effect of perhaps forcing the enemy to make suboptimal decisions in the face of incoming hostile fleets. Do they react with the exact amount of ships needed? Do they not do certain actions?

A smaller part of my offence was directly attacking enemy fleet to either take their mined kore or deal double damage. The only thing of note I added here was to incorporate a score measurement for attacks to instead give preference to mining routes which were deemed more efficient. Sometimes I would add a feature or fix a bug here and it would actually do worse.
- I had an off by one issue where sometimes a shipyard wouldn't attack a fleet even if it could, but fixing that made it attack too often without much gain.
- I experimented with attacking enemy fleets that were converting into shipyards. I never really saw this activate much, and it was coupled with the previous failed bug fix so I ended up scrapping it.

### Expansion
Ah... my worst nightmare. In my opinion, deciding when and where to make new shipyards is often the most important part of the game. It's a delicate balance between greed and safety. On one hand, expanding early gives you quick access to tiles that were seldom harvested, giving you a surge of kore which can snowball into a lead. On the other hand, creating a new shipyard puts you at a 50 ship disadvantage, so if you can't defend against a quick attack you're out of luck.

As for when to expand, I mostly kept aDg4b's logic taking into account the current shipyard spawning capacity and fleet distance. If the current kore was significantly more than what I could spawn before most of my fleets returned, I would make a shipyard. The hardest part of tweaking this was mitigating situations where I would expand too early or late, or make multiple shipyards in a row. To be honest, I didn't find a great solution for this. I first tried enforcing minimum time constraints between building shipyards, but when that didn't work well I settled with lower bounding the perceived fleet distance and spawning capacity for each shipyard. I also established some minimum and maximum constraints on ship to shipyard ratios when expanding.

Choosing expansion spots is another challenge. Here's a list of each of the factors that I took into account to rank each tile on the game board. Some of them were inspired from chess principles.

**Kore:** I evaluated kore at nearby tiles and summed them up, using this as the driving force behind the value of a particular expansion spot. I modified the perceived kore on a tile based on a few things:
- Kore is raised to the power of 1.1, serving to give bonus to large kore stockpiles that will give quick turnover after the expansion. This helps mitigate the immediate 50 ship penalty when defending a quick attack.
- A gaussian distribution decreasing the value of a tile based on the distance from the new chosen shipyard position. I tried out different distributions here that might synergize with the mining routes I expected to launch, but they didn't seem to help much.
- A linear penalty decreasing the value of tiles close to other friendly shipyards. This avoids creating shipyards too close together.
- A bonus multiplicative factor that acted to incentivize taking space away from the opponent. Tiles that were hotly contested (read: were roughly equidistant from enemy and friendly shipyards) were given a bonus if the new spot decreased the distance of the nearest friendly shipyard.

The kore sum was then normalized so that the following penalties were not outshadowed.

**Penalties:** As important as it is for there to be a lot of nearby kore, it's also important to not overextend into enemy territory.
- Distance penalties were given based on both the average distance from all friendly shipyards, and the distance to the closest friendly shipyard.
- An enemy penalty measuring the danger of a spot by considering the size of a fleet that the enemy could send in the near future. The enemy fleet size was also compared to the amount of reinforcements I could send in reaction to it, with extra penalty added the higher the difference was. I spent a lot of time tweaking this and thinking about things like normalization and growth speed. Ultimately, I used a linear factor based on the difference between the distances of the closest enemy and friendly shipyards. I also put a soft cap on this penalty that grew with the size of the enemy fleet.

Once an expansion was triggered, it continued until either it finished, or some exceptional situation caused it to abort, like going under a minimum number of ships or the enemy beginning to stockpile ships for an attack.

### Mining
A quick aside: Before a shipyard decides to mine, it checks if we have a kore surplus and are behind in ships. If the shipyard already has some fleets out and has a significantly smaller than the average number of ships per shipyard, it prefers to spawn instead.

Choosing what mining routes to send and when was another one of the most important things to think about. I'll split this section into two parts: route generation and route selection.

**Route generation:**
Route generation started off simple: choose a point on the board and the closest shipyard and consider all four routes consisting of "L" shaped paths between them. I had an attempt at creating more complex mining routes by considering a second nearby point before the destination, but it often increased the computation by a significant amount without much benefit, so it remains deactivated. By the end, I ended up hard coding in some new route patterns inspired by other competitors' routes.

Close to the end of the competition, I started to think more about what destinations a mining route should be sent to. Instead of just the closest shipyard, I would sometimes force all mining routes to go to shipyards that needed more ships, like a newly created one or one that was under attack.

If I had more time, I would have spent more time thinking about destinations, as distributing your ships well is a great step towards reacting to enemy threats better. I tried an unsuccessful allocation strategy which would try to keep the number of ships at a shipyard close to a specified target, which was based on its kore production and its danger relative to the size of potential enemy attacks.

**Route selection**
Fleet size was chosen based on how quickly my kore reserve could be depleted. With not much kore, it's less important to spawn, so I would send the minimum fleet size, only increasing it if an enemy could launch a fleet to intercept it. Otherwise, I would send all of the available ships. Tieing up a large number of ships in a single fleet or wasting a turn on a really short route are both not good outcomes. As such, I also established some minimum and maximum constraints on fleet and route sizes based on available or total ships. Moreover, an important late addition was to not send large fleets if it resulted in a nearby enemy shipyard being able to launch an attack.

I scored routes based on the average kore obtained per turn, except I (mostly) ignored collection rates. This results in a balance of sending more efficient small routes versus more kore-producing long routes. I experimented with some other bonuses and penalties to routes as well. For example, I penalized routes based on the number of turns before it came back where the shipyard could neither spawn nor launch a new fleet. This often had the effect of a shipyard sending routes that were 1 or 2 tiles shorter than the previous one, resulting in a chain of routes that come back at roughly the same time. The hope was that, when I eventually decide to make a new shipyard, I spend less time waiting to launch.

I would then launch the highest scoring route, assuming I could launch it and that it passed some other checks (e.g. I wouldn't launch small fleets from shipyards that had better than average spawning production if I had a surplus of kore). Here I also experimented with launching a smaller intermediate route if I could still launch the highest scoring one in the same amount of time, but it didn't perform as well as I would have liked. I think it had a side effect of increasing the average time to launch larger routes, resulting in less kore in the future.

## Conclusion
I thoroughly enjoyed this competition. It turned out to be a lot deeper than I initially thought and learned a ton from it and the other competitors. Again, a thanks to the Kaggle team for a great competition. I look forward to future ones! 

For anybody that wants to take a look at my agent, here is a link to my [code](https://github.com/itswin/Kore-2022).
