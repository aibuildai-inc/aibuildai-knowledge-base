# 5th place short notes

Competition: abstraction-and-reasoning-challenge
Rank: #5
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154377

Congratulations to winners! And thanks to organizers - this was very interesting and unique competition.

I'll leave longer descriptions to Top 3 teams, will just write some very brief summary of my approach ;)

In general my idea is to use 3 connected functions:

- A: cut parts from input by color/shape/location/etc;
- B: re-color/modify/rotate/etc parts from step A output;
- C: combine parts from step B output in different ways, like AND, OR, XOR, hstack, vstack and many others and check if this gives valid solution.

In each step "no changes" is also a valid result, in that way such pipeline could solve also tasks where e.g. only cropping is needed without further changes and ensambling parts together or where only re-coloring is needed.

This is pretty much like brute-force and as there are a lot of possible outcomes in each step, multiplication of all variations of all 3 steps together was huge - but here helped the fact that public=private, so I used a lot of "feedback" from submissions to tune my functions and delete what wasn't helping.

See you in future competitions!
