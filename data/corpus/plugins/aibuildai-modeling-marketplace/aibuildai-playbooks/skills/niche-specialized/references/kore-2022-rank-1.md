# 1st place solution

Competition: kore-2022
Rank: #1
Source: https://www.kaggle.com/c/kore-2022/discussion/340035

# The competition
Thanks @bovard and all fellow competitors for the great fun during this interesting competition. I joined this competition to get into reinforcement learning, but while playing around with the great Kore Beta 1st place solution by @egrehbbt I managed to get up in the ranks quickly, mainly by changing the expansion strategy. I decided to focus on the rule-based approach and see how far it would get me. 

My approach was a bit how one would develop a software product. I had a list of issues, kept all code under version control and did iterations solving a few issues and bugs at a time. For this the excellent visualization tool by @jmerle was vital. It provided me with all overviews I needed to identify why I lost a certain episode.

When I joined the competition it was dominated by @shuntarotanaka and @qihuaz. At some point I managed to get to 3rd spot after which I dropped again to 6th or 7th. By then 1Musketeer @itswin jumped towards #1, a position which he managed to keep for weeks mid competition. His approach was very powerful and hard to beat. Thanks @itswin for sharing your agent with us.

At some point I managed to get to the #1 position with my V92 agent. This agent kept that position for a few weeks. I actually had a very hard time beating it myself, even though I made many changes that I thought would be beneficial for sure. It took me two weeks and 26 updates to get agent V118 that was consistently better in 1v1 matches. That agent turned out to also do well against other competitors and the lead that I managed to acquire was astonishing, about 200 points or so compared to #2. 

Then on the second day after submission end I looked into the gaze of the basilisk @basilisk1337. I saw a match that caught me off-guard against one of his last day submissions. It turned out he found a way to paralize my first shipyard, making me unable to expand. The fix would have been straightforward, but I guess it was the beauty of the basilisk gaze that I could not even make my agent look away. After that match basilisk quickly collected points from all my agents, as they shared the weakness and quickly he rose to #1. Then it seemed a game of chance, one day he was #1, then the other I was #1. The convergence of the ranking seems to be a bit unstable. I have some thoughts, but can share those in a different post.

Then on the last day of match execution my V127abandonrescuecollideattack (version 127/146) managed to get the lead in this competition. It was actually the last version where I made changes in the rules itself. All subsequent versions were variations in the value of the parameters to tune the rules. To get to the winning agent was a lot of coding and a lot of trial and error, a combination of introducing new rules and tuning of parameters.

That said, I learned a lot during this competition:
•	I ran into timing problems, requiring me to optimize route search and apply more caching
•	I ran into memory problems, requiring me to dive into cache management
•	At some point I got an account the Github CoPilot, which improved my coding productivity considerably (particularly for the parts of code that one hates to write, such as print statements)
•	Improved my debugging and general Python coding skills

This competition was great fun, thanks to all of my fellow competitors. Perhaps our agents will meet again in a future simulation competition!

# The winning agent
My agent is based on the Kore Beta 1st solution by @egrehbbt and ported by @realneuralnetwork. While one will probably recognize a lot, most of the sections were adapted. According to Kaggle I added 5923 lines and removed 777.

The agent conceptualizes the following action phases, as inherited from egrehbbts solution:
•	Shipyard defense
•	Fleet attacks: adjacent and direct attacks
•	Shipyard attacks
•	Expansion
•	Mining
•	Spawning

I largely kept this framework in place, but added two large concepts: 
•	Board state: 
For each point on the board, for each point in time the agent keeps a ledger of the actual damage and the damage potential of both myself and my opponent.
•	Routing:
Given an objective the agent calculates the score of (almost all) potential routes that could be chosen

Applied to the agents phases, along with some additional tweaks:
•	Shipyard defense: I calculate whether or not I am able to defend an attack. If so, I spawn and send reinforcements from other shipyards. If I cannot defend, I added an abandon pattern, where my fleet leaves the shipyard undefended and tries to: attack another shipyard, rescue incoming fleets, attack smaller attacking fleets, jump to another shipyard, or just leave the shipyard and attack it after its loss a few turns later.
•	Fleet attacks: I split up adjacent attacks and direct attacks due to the difference in survival. An adjacent attack is intended to die, so it had a different scoring criterium. For direct attacks it would be more or less a mining route with an attack included. I spent a lot of time managing both the execution and prevention of direct attacks, which helped me to create a good representation of the board state.
•	Shipyard attacks: rather than collecting all ships at one shipyard to create a maurading fleet, I decided to use a distributed approach. I calculate whether I can combine the fleets from different shipyards to beat an enemy shipyard. At first I had a hard time beating musketeer, who had an excellent adjacent attack pattern, but then I managed to route the fleets to minimize adjacent attack risk.
•	Expansion: this is where I got the quickest gains compared to the Beta 1st solution. My spot picking was a balance between kore, crowding and damage positions. I tried several parameters here, usually tweaking at least one with every agent submitted.
•	Mining: this has been core in my strategy, though turned out to be tricky. I chose to maximize mining efficiency per ship per tick. In fact, the complex routes that I saw @shuntarotanaka use prompted me to introduce the routing approach that could try almost all possible routes. This helps me in maximizing kore early game. What I do have a bit of a problem with is that my mining is a bit like a swarm of locusts, mining kore so fast that the kore spot is depleted and more or less useless afterwards. I saw others preserving their nearby kore spots in a better way. So I needed to match this strategy with aggressive play to leverage the fast mining of kore.
•	Spawning: the agent usually spawns if it doesn’t mine or if it has plenty of kore

Even though my agent is rule-based, it is a collection of patterns which turned out to behave in ways I did not anticipate at first. My shipyard attack patterns, though designed for late game, were for example quite effective in taking over too risky first new enemy shipyards.

All in all, I had great fun!

Thanks all for reading!
