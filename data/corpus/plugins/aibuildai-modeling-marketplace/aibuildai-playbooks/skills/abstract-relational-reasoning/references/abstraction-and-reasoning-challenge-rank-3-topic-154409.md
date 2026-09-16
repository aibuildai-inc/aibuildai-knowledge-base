# My part of the 3rd place solution

Competition: abstraction-and-reasoning-challenge
Rank: #3
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154409

Here is the description of our final solution https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154305
@golubev has definitely contributed more than I into the final result, but even my solo solution could get at least the high silver so I share it with all the source code.

The code and more detailed approach description can be found here: https://github.com/IliaLarchenko/abstract_reasoning

Below is the short write up of my approach.
 
- In general, my method can be described as a domain-specific language (DSL) based. I have created the abstract representation of the colors, image blocks extracted from the original input image, and different binary masks. Using these abstractions, I wrote several "predictors" that try to derive a set of transformations used to get every output image from the corresponding input image and, if it succeeds – apply these transformations to test input to get the final answer.

- By the moment of team-up, I had 5 tasks solved (+2 from public kernels). ([my kernel demonstrating how it works](https://www.kaggle.com/ilialar/5-tasks-part-of-3rd-place-solution)). Probably there are other tasks that can be solved as well, but I have not tested my solution separately after the merge. My approach also solves 138 and 96 samples from the train and valid sets, respectively.

- The general logic of every predictor is described in the pseudo-code below (although, it can be different for some classes).
```
for n, (input_image, output_image) in enumerate(sample[‘train’]):
    list_of_solutions = []
    for possible_solution in all_possible_solutions:
        if apply_solution(input_image, possible_solution) == output_image:
            list_of_solutions.append(possible_solution)
    if n == 0:
        final_list_of_solutions = list_of_solutions
    else:
        final_list_of_solutions = intersection(list_of_solutions, final_list_of_solutions)

    if len(final_list_of_solutions) == 0
        return None

answers = []
for test_input_image in sample[‘test’]:
    answers.append([])
    for solution in final_list_of_solutions:
        answers[-1].append(apply_solution(test_input_image, solution))

return answers
```

- The best working predictors (on the test set) were those that restore some kind of mosaic on images.

- Other important tips:
– I have used multiprocessing to significantly speed up computations and control RAM and time each process uses.
– I have cached all blocks generated at any step. Without it, 16gb RAM often was not enough.
– I have used unit test extensively to control that I have not broken anything with my last changes (I think, it saved me a lot of hours)
