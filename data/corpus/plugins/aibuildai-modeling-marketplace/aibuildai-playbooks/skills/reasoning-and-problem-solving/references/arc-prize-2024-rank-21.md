# 21st place: a short report

Competition: arc-prize-2024
Rank: #21
Source: https://www.kaggle.com/c/arc-prize-2024/discussion/550209

Hi everyone! I would like to share my 28% "solution", even though there is not a lot to share. It's a very minor insight, and around 1% of the effort (the rest was a failure). The idea is not very original, just some recoloring as a preprocessing step for the icecuber's solver. In the ensemble, it adds +2% above the 26% baseline. This is the function:
```
def recolor_task(task):
    '''Recoloring effort via pixel counting (but it ruins some tasks)'''
    if len(task['test']) > 1:
        return

    all_a = []
    all_b = []
    for pair in task['train']:
        input_colors, input_counts = np.unique(pair['input'], return_counts=True)
        a = list(zip(input_colors.tolist(), input_counts))
        a = tuple(x[0] for x in sorted(a, key=lambda x: x[1], reverse=True))
        all_a.append(a)
        all_b.append(tuple(np.unique(pair['output']).tolist()))
    uni_a = set(sum(all_a, ()))
    uni_b = set(sum(all_b, ()))

    input_colors, input_counts = np.unique(task['test'][0]['input'], return_counts=True)
    c = list(zip(input_colors.tolist(), input_counts))
    c = [x[0] for x in sorted(c, key=lambda x: x[1], reverse=True)]

    if uni_a == set(c) or len(list(uni_a & uni_b)) == 0:
        return

    for i in range(len(task['train'])):
        raw = all_a[i]
        ti = task['train'][i]['input']
        to = task['train'][i]['output']
        for j in range(len(c)):
            if j == len(raw):
                break
            ti = [[c[j] + 10 if x == raw[j] else x for x in sub] for sub in ti]
            to = [[c[j] + 10 if x == raw[j] else x for x in sub] for sub in to]
        ti = [[x - 10 if x > 9 else x for x in sub] for sub in ti]
        to = [[x - 10 if x > 9 else x for x in sub] for sub in to]
        task['train'][i]['input'] = ti
        task['train'][i]['output'] = to
```
I didn't test this particular version, but it should work. It can be improved, but in this case I think it doesn't matter, as the score stays the same or even degrades. Or maybe I didn't explore enough. But, again, there is no merit in it, it doesn't get you much closer to AGI algorithms.

So the insight is about looking at the task and being able almost immediately tell if the coloring is irrelevant. I suppose this is related to what Guillermo Barbadillo [said](https://www.kaggle.com/competitions/arc-prize-2024/discussion/545671) about representations. Irrelevant perspectives are quickly discarded. For example ([1e32b0e9](https://arcprize.org/play?task=1e32b0e9)), this is how the task looks like:

And this is how I actually see it conceptually after just a moment, before understanding the solution:

Or something like that, I can be wrong, of course, but it kind of helped me, so who knows.

I was trying to think big, but I lack both skill and creativity for that. And motivation. Sometimes I can work, sometimes I cannot... Anyway, I ended up exploring the [graph approach](https://www.kaggle.com/competitions/arc-prize-2024/discussion/518930#3064762), but with no success. I also tried neural cellular automata, but couldn't go beyond 1%.

I decided not to touch LLMs. Why? I knew that many others would do it, so there is no point to pursue the same thing. And I still don't understand the transformer architecture on the math level, it would be nearly impossible for me to go beyond already existing stuff. Plus, I have limited compute. I also decided not to tinker with the icecuber's solver. Why? My C++ skills are nearly nonexistent. I would have to learn the language first. And, of course, this approach doesn't promise any AGI, so what's the point.

Overall, I really like this whole quest for AGI. Maybe my naive fascination is not enough for a meaningful contribution, but I really appreciate the opportunity, so thanks to everyone who made this all possible. Let's see what happens next.
