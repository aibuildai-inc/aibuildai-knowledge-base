# 20th place solution

Competition: abstraction-and-reasoning-challenge
Rank: #19
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154308

I guess you are interested in TOP solutions, yeah me too!  
But please let me write here, as I spent mush energy for this challenge.  

My Solution is DSL plus search.   
https://github.com/k-harada/abstraction-and-reasoning-challenge  
My DSL applied like this.  

```
def test_58():
    p = Runner(58, verbose=True)
    p.run("mesh_split")
    p.run("n_cell")
    p.run("keep_max")
    p.run("max_color")
    p.run("paste_color_full")
```

I tried some kinds of search, and simple BFS significantly improved my score, 0.970 -&gt; 0.931.  
No idea for probing.  

Solved: 
Train: 146 / 416  
Evaluation: 94 / 419
