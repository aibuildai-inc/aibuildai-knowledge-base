# 11th solution

Competition: lux-ai-2021
Rank: #11
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/296306

Firstly, special thank to @stonet2000 for hosting such an amazing competition! Thanks @shoheiazuma for the wonderful public kernel and @zaharch for the UNet approach sharing. And many other people who shared ideas and insights... You guys did really good job!

My solution is based on @shoheiazuma 's public kernel [https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning](url). (THANKS AGAIN!) I list the modification I did below for your convenience to read. I will also mark those making my bots better than the original kernel and @shoheiazuma 's best kernel ;)

* bto = better than original one; btb = better than the sazuma's best bot

1. Center action (bto)
2. SE block (bto; btb)
3. Single Unit level network (probably btb)
4. Learn City actions: Stay, Build worker and research (bto)
5. more data (~2300 episodes) (bto; btb)
6. Learn 2nd place bots as well (bto; btb)
7. Expand the channels in the hidden layers (bto; btb)

Here I list things that might be useful but I did not implement below:

1. Use three-by-three grids as output head ( I can do it since i use unit-level policy, this can keep the spatial information)
2. Ensemble with UNet model (add some global information to consider)
3. Add some RULES to avoid some stupid moves (that is because imitation learning's overfitting)

Finally, insights that I suppose but might be wrong:

1. For imitation learning, all the SAME settings, unit-level policy outperforms the map-level policy. But they should handle the performance-time trade-off.
2. If you want to do the transfer learning (from imitating to reinforce learning), you should use map-level policy. That is because RL performs better with map-level policy.

Thank you for your reading! Hope to see you guys in Season 2!
