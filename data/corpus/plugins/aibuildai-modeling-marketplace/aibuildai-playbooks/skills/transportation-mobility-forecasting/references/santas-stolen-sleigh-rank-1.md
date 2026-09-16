# Winning entry

Competition: santas-stolen-sleigh
Rank: #1
Source: https://www.kaggle.com/c/santas-stolen-sleigh/discussion/18312

Hi! I share some of materials we used in the contest.

In the first phase, we used SA, using 20 threads in 24-core machine, running ~5 days. Ran in 3 machines and best score was 12384669985.

After that, BS combined with SA, is used, again in 3 machines, this time running 21 hours. Best output was 12384529598. Some complicated merging of these output was our winning entry, 12384507107.

Below is a tracking sp of our progression, after we made a team.

https://docs.google.com/spreadsheets/d/1bO6zLWm3JMOa8rvBhxpV688bUndeJUo595Uynv1RQY8/edit?usp=sharing

Code for the first phase is attached. Download them in a same directory. sort_by_dist is just a list of nearest presents, encoded in a text. prv.csv can be any output file, possibly your submission.

g++ santa_mul.cc -o santa_mul -pthread --std=c++11 -Ofast -march=native

(./santa_mul output_file output_file_with_more_information progression_output_file number_of_iterations starting_temperature number_of_threads)

time ./santa_mul o3.out d3.out p3.out 500000000000 0.33 20
