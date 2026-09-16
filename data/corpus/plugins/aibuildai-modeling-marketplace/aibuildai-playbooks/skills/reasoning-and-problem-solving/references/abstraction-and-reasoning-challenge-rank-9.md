# 9th place solution(s)

Competition: abstraction-and-reasoning-challenge
Rank: #9
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154319

It's fantastic to see the huge improvement of all teams in the last week of this interesting competition. Like many others, I and my teammates have had a sleepless night to play with the last "tickets." 

Here is a short list to sum up our team solutions:

- 2 DSL: 8/102
- 2 Pub: 2/102
- CA: 2/102
- CA + voting: 1/102
- DSL + alignment: 1/102

---
@user189546's DSL and CA, and her own words

My part of our team solution scores 7 points on LB. 
It consists of 2 approaches:
- Crop object + symmetry transformations (5 points). It overlapped by 2 tasks with @yukikubo123 DSL+search solution.
- Cellular Automata recurrent CNN public kernel + encoding colors by frequency (2 points, however first time it scored 3 points, didn’t happen again).

1) Crop object

This competition is hard for many reasons and one of reasons is large output space. When I joined the competition  I wanted to get first small win quickly. So  I looked at tasks that have small output space – crop object tasks. First object crop kernel scored 2 points and when I added more object extraction functions and symmetry transformations it scored 5 points.

Kernel uses brute force to find right object extraction function and right symmetry transformation. If number of possible unique crop candidates &gt; 3 it predicts 3 biggest objects. Small objects can be features or noise. Solution runs in 3 min in Python.

*Pseudocode:*

```
bg_color = detect_bg(task) # find background color as color of figure that touches 3 sides of input.
get_object_funcs = [get_objects_by_color, get_objects_by_connectivity, get_objects_by_color_and_connectivity, get_objects_rectangles_without_noise, ….]

symmetry_funcs = [identity, flip, rotate, tile, zoom, flip_diagonal, …]

for get_object_func in get_object_funcs:
	for symmetry_func in symmetry_funcs:
		train_prediction_candidates =  symmetry_func(get_object_func( task_train_examples))
		if task_train_outputs in crop_candidates:
			break
test_prediction_candidates = symmetry_func(get_object_func(task_test_examples))
test_predictions = unique(sort_by_size(test_prediction_candidates))[:3]
```

[**Sample Notebook**](https://www.kaggle.com/user189546/5-crop-tasks-by-brute-force)

2) CA + ColorFreqEncoder

Thanks to @teddykoker we had wonderful CA kernel: https://www.kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks. It solved plenty of various tasks in train and eval however didn’t score on LB. I added encoding colors by frequency (0 – most common color, 1 – second most common color, 2 - …) and fixed all random seeds. Kernel performance is highly random so without random seeds it is not reproducible. When I submitted first time without fixing all random seeds it scored 0.97, however when I fixed random seeds it scored 0.98.

[**Sample Notebook**](https://www.kaggle.com/user189546/cellular-automata-learning-with-colorfreqencoder)

---
@yukikubo123's DSL

I won't post the detail of this solution here because @yukikubo123's GitLab [*repo*](https://gitlab.com/yuki1020/abstraction-and-reasoning-challenge/) alone is enough to show all of his awesome work of art
-&gt; https://gitlab.com/yuki1020/abstraction-and-reasoning-challenge/

---
@tarobxl's alignment technique

When observing the data, @tarobxl has thought: “Could we ease the learning process of the model(s), simply by rotating the training pairs of a task in the same direction.” I and he have applied this idea to all available solutions/models and luckily it works with @yukikubo123’s DSL and gives us a new solved task in the LB.

[**Sample Notebook**](https://www.kaggle.com/phunghieu/yukikubo123-s-dsl-with-alignment)

[img_1]

---
@phunghieu's notes [Topic Author]
- What I’ve tried
  - Try using CNN to solve ARC’s problem in Deep Learning approach -&gt; failed
  - Try merging @yukikubo123’s and @user189546’s solutions with 2 public kernels to yield the 0.931 scored solution
  - Successfully optimize the CA solution to run faster (x2) on GPU accelerator by removing redundancy, duplicate jobs, and trying to use batch-size &gt; 1 whenever possible
  - Try averaging ensemble technique for CA solution -&gt; failed
  - (I &amp; @tarobxl) Successfully use the voting mechanism to ensemble CA kernel -&gt; 1 more solved task
  - Try to add augmentation to CA solution -&gt; failed
  - Successfully apply the alignment technique for @yukikubo123’s DSL solution -&gt; 1 more solved task

- What I’ve not tried
  - Modify the CA solution to solve tasks that have differences between the input’s size and output’s size
  - Use multi-processing to further improve the CA solution performance, thus we could try more idea within 2 hours limit of a submission uses accelerator (GPU)

- My main contributions to the team are
  - Merging all solutions into a big solver
  - Improving the performance of all available solutions to fit them into a time-limited submission

(*) A few last words: Although not being a big brainer to the team, I have, somehow, tried my best to make use of my programming skills and techniques which I frequently used in the Deep Learning field to support my teammates. I also have taken advice from @tarobxl and @ysharma1126 to accelerate my “solution-optimization” job.

---
Happy Kaggling!!!
