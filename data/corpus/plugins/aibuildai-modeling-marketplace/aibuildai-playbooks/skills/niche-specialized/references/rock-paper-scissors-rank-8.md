# 8th place solution

Competition: rock-paper-scissors
Rank: #8
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221500

It's the first time for me to get a gold medal🎉
Thank everyone participating in this competition. 
In this competition many players share their own thoughts and solutions. 
I learned a lot from them.

  
This is my notebook which got gold medal.
https://www.kaggle.com/enukuro/rps-8th-place-solution

## First attempt
In early period I first tried to use neural network learning top players strategy but it's completly failed.


## Developing baseline
Next I read discussions and decided to follow thier idea.
I thought [Going meta with Kumoko](https://www.kaggle.com/chankhavu/going-meta-with-kumoko) is good starting point and started to develop my own baseline agent like Kumoko.
  
I knew ensemble is the key, so wanted to integrate many algorithms but noticed it's very tiring work...
I'm not good at dealing with variable scope problems in Python😅
So I needed to invent some solutions for this problem.
This is my solution.
load, rewrite and save file.
In this way I can keep variable scope clean and easily integrated any algorithms😄


```
with open("/kaggle_simulations/agent/rps-dojo-data/black_belt/centrifugal_bumblepuppy_v4.py") as f:
    s = f.read()
    s = s + "\n\ndef set_output(_output):\n    global gg\n    gg['output'] = _output\n\n"         
with open(BASE_PATH+"centrifugal_bumblepuppy_v4.py", mode='w') as f:
    f.write(s)
with open(BASE_PATH+"centrifugal_bumblepuppy_v4_mirror.py", mode='w') as f:
    f.write(s)
    
import centrifugal_bumblepuppy_v4
import centrifugal_bumblepuppy_v4_mirror

def centrifugal_bumblepuppy_v4_agent(observation, configuration, my_last_action):
    if observation.step == 0:
        importlib.reload(centrifugal_bumblepuppy_v4)
    else:
        centrifugal_bumblepuppy_v4.set_output(['R','P','S'][my_last_action])    

    return centrifugal_bumblepuppy_v4.run(observation, configuration)

def centrifugal_bumblepuppy_v4_mirror_agent(observation, configuration, my_last_action):
    if observation.step == 0:
        importlib.reload(centrifugal_bumblepuppy_v4_mirror)
        observation_mirror = observation
    else:
        observation_mirror = {"step": observation.step, "lastOpponentAction": my_last_action}
        centrifugal_bumblepuppy_v4_mirror.set_output(['R','P','S'][observation.lastOpponentAction])    

    return BEAT[centrifugal_bumblepuppy_v4_mirror.run(observation, configuration)]
```


Other parts are not so special.

## Action Selection

Weighting by dllu score and random.choices from them.

`best_index = random.choices(range(len(scores)),weights=([max(0, score) for score in scores]))[0]`

## One Tweak

Changing strategy by score history.
In worse situation it uses more random.
But I'm not sure it's effective or not.
Before evaluation period, this strategy didn't seem to work well so I didn't explore much. 
Probably many other players also tried strategy like this.

```
    if observation.step > 0 and strategy_type == 0 and sum(score_history[-strategy_assess_length:]) < -strategy_change_point:
        strategy_type = 1
    if  strategy_type == 1:
        my_action = BEAT[my_action]
        if random_strategy and random.random() < 0.5:
            my_action = random.randint(0, 2)
        strategy1_count += 1
    if strategy1_count > strategy1_count_length:
        strategy_type = 2
        strategy1_count = 0
    if strategy_type == 2:
        strategy2_count += 1
        my_action = random.randint(0, 2)  
    if strategy2_count > strategy2_count_length:
        strategy_type = 0
        strategy2_count = 0
    if strategy_type == 3:
        my_action = random.randint(0, 2)  
```

## Conclusion

In later period, my agents in middle period all dropped out probably because of [RPS Geometry🦇](https://www.kaggle.com/superant/rps-geometry-silver-rank-by-minimal-logic)？
So I had to develop new strategy.
Later period I also could get good results, some agents reached to gold zone, but in evaluation period they all dropped out...
Finally my top agent in gold zone is very old one before appearance of RPS Geometry🦇.
This agent hadn't reached score 900 before evaluation period.
   
My later period notebook is like this.
https://www.kaggle.com/enukuro/rps-part-of-the-near-gold-zone-solution 
  
I almost never submitted the same params agent, never stick to one approach.
For me rankings in before evaluation period was not reliable so agents diversity was important? 
  
Anyway I'm very happy that I proved my luck🥳
