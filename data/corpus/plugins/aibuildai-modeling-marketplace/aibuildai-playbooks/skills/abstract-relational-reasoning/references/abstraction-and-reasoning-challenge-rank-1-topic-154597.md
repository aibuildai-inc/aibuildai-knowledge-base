# 1st place solution + code and official documentation

Competition: abstraction-and-reasoning-challenge
Rank: #1
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154597

Thanks to @fchollet for making this dataset, and organizing the competition. I think understanding and making intelligence is extremely interesting, and working on this problem and reading the dataset's paper significantly changed my view on AGI. The problem of squeezing prior knowledge and generalization into your model really becomes clear when you need to do it yourself. I had actually read the paper before the contest was announced, but working on the contest gave me a much more detailed view of the practical side of things.

Unfortunately, I don't feel like my solution itself brings us closer to AGI. The main component is a DSL which applies up to 4 of 142 unary transformations (42 different functions, some have different variants). This DSL is solved by enumeration (exploiting duplicates) + a greedy stacking combiner. Everything is implemented efficiently in C++ (with no dependencies) and running in parallel.

My code is currently &gt; 10k lines long, but (I think) most of it is not used. I will publish it after removing the unused / deprecated / failed code.

Edit: Code and official documentation now available at
[https://github.com/top-quarks/ARC-solution](https://github.com/top-quarks/ARC-solution) and 
[https://github.com/top-quarks/ARC-solution/blob/master/ARC-solution_documentation.pdf
](https://github.com/top-quarks/ARC-solution/blob/master/ARC-solution_documentation.pdf)

Edit 2: Public notebook scoring 0.794
[https://www.kaggle.com/icecuber/arc-1st-place-solution
](https://www.kaggle.com/icecuber/arc-1st-place-solution)

###My path


In practice I started out with implementing the first 100 training tasks by hand, extracting and generalizing useful functions from that. I made a brute force combiner of these transformations, choosing to only support transformations from images to a single image. Originally I wanted more types such as positions and colors, but it turned out that it was easier to represent these as images (such as a dot at the position, or a dot of a given color). This decision made it much easier to make an DSL, and got 10 tasks on the LB.

I then noticed that distribution of training, evaluation and LB were quite different, so I decided the evaluation dataset was better used as training data. I hand-coded 100 evaluation tasks, which I used to add more transformations, and improve my DSL. I noticed that functions taking more than 1 argument were growing my search space super-exponentially, and that they usually just took either the input or output image size as second argument. This led me to only keep unary functions. I also added lists of images as a type, to solve tasks that required looping. I also started representing my DSL in a DAG to exploit duplicates (multiple transformations giving same output). This required me to rewrite most of my code, but gave me 14 tasks on LB.

Optimizations, multithreading, reducing memory usage, etc. gave me 17 on LB. And augmenting my samples with diagonally flipped tasks, boosted me up to 21.

Note that the above (and below) is simplified, and there are many, many details I'm forgetting / omitting as I write this.


## Main solver steps
### 1. Recolor inputs
I remap colors in each sample input to be consistent across samples. A typical use of this is when each sample has a single color, but it is different for each sample. The recoloring would make all of them equal. I do this through iteratively, greedily recoloring the most extreme color across all images using features such as size, number of objects or position. So if in each sample input there is a single-colored object considerably larger than all others, I make all those objects the same color. This works surprisingly well, even for cases I did not think I accounted for. It even straight up solves some easy tasks from the training set by making all outputs equal.

### 2. Maybe add diagonally flipped images
I double the sample inputs by adding the same tasks flipped along a diagonal. This makes for 3 full runs (depth 3 for performance reasons): one normal, and one for each diagonal. This simple addition moved me from 17 to 21 tasks solved on the LB (and hence helped more than all the optimizations needed for doing depth 4 search instead of depth 3).

### 3. Solve for output size
I first generate pieces as explained in "Overview of DSL". I do combinations and transformations of piece sizes to fit the training samples, and pick the one consistent with the most training samples. I break ties by approximately ordering the combinations by complexity (so transforming a piece with size WxH to get (2W)x(H+1) is more likely than (W+2)x(3H). The output size is correct for about 97% of the evaluation tasks.

### 4. Generate transformed pieces using output size
Details in "Overview of DSL". Output size is used to give the second argument to functions like embed, which crops / pads images to a given size.

### 5. Greedily stack pieces to fit a training sample
See "Overview of DSL".

### 6. Pick the top 3 candidates
I order candidates by the following criteria in decreasing order of importance:
 - Fits most training samples
 - Uses fewest transformations (max depth across pieces)
 - Uses fewest pieces in the stack



## Overview of DSL
All the nodes in my DSL are either images, or lists of images. An image has a position, a size, and a 2d array of colors (0-9). Black (0) is treated as transparent in most of the functions.

First, sequentially apply up to 3 or 4 unary transformations (see "List of transformations") to the input image. Then optionally apply a subset of the following (in order):
 - Stack lists of images
 - Move image to origin
 - Color all non-black pixels some color (for each color in every training output)
 - Resize image (crop / pad) to fit output size

I call result of the above transformations "pieces". Finally, stack some pieces to produce a final output.


My solver mainly enumerates all the pieces, and applies a greedy solver to stack them. Since there are 142 image transformations at each depth, plus transformations at the end, we can't naively generate all the pieces. Fortunately, many different paths of transformations lead to the same image. I therefore make a DAG (Directed Acyclic Graph) for each training input, where each node is a image, and edges are transformations. This typically leaves on the order of 10^7 nodes in the DAGs and a few million unique pieces (where pieces are different if they differ for any sample input image or the test image).

I run multiple tasks in parallel (4 at depth 3, 2 at depth 4) using a script which tries to schedule all the tasks to fit within memory and 9 hour time limit, this runs maybe 50% of the images at depth 4 and the rest at depth 3 (I didn't really benchmark it, except 67%  were run at depth 4 in an old version). The memory limit was typically more restrictive than the time limit, so I reduced memory usage through several tricks, such as huffman coding the images in the DAGs and using a custom memory-efficient hashmap.

My greedy stacker produces each candidate output as follows:
 - Split pieces into black parts and non-black parts (parts not picked are treated as transparent)
 - Pick a guiding sample
 - Pick some subset of samples to "care about"
 - Up to 10 times:
  - Among all the pieces which do not contradict any of the images we care about
    - Add the piece which fits the most new pixels in the guiding image
 - If the test output image contains no holes, add this as a candidate

The stacker is optimized using bitsets to be fast and memory efficient.



##Some notes about choices and failed ideas:

Greedily stacking outputs worked surprisingly well, even solving many tasks I didn't think it could solve. It consistently beat all my attempts at alternatives and improvements, so it just stayed mostly the same from the beginning.

I tried several ideas which didn't make it. One of these was adding general functions of more than one variable. However, none of my many attempts seemed to be able to give a significant improvement.

Another idea was a system of deducing transformations from the training outputs, which I really wanted to work. If I could make this in a general way, I could maybe apply meet-in-the-middle to search double the depth! However, after many attempts I found that deductions are way harder to stack and generalize than transformations. This made it much harder to generalize deductions, and ultimately only a single deduction was kept (deducing "kronecker products" as last step).

Yet another approach was generating a large number of sample tasks to train ML algorithms and fit hyperparameters (f.ex. different costs for each transformation). However, I concluded that generating new tasks was harder than solving them.

I always felt ranking candidates was a much easier problem than generating candidates (f.ex. if all inputs are symmetrical, then output should probably be symmetrical). Ideally this could be used to fill in parts of the image, or generate a huge number of candidates and leave most of the work to the ranking algorithm. However, I never found a way to exploit powerful ranking to my advantage. My simple ranking heuristics worked well enough for its purpose.


##Other

###Statistics for the evaluation dataset
Depth 3, without adding flipped images. Faster, but probably worse than a full run.

Total:   419 (Some of the 400 tasks have multiple tests)
Size :   409 (97.6% deduced the correct output size)
Cands:   176 (42% had the correct answer among the generated candidates)
Correct: 164 (39% gave the correct answer among top 3 answers)



###List of transformations
Many functions have many variations, but are only listed once (f.ex. there are 14 variations of pickMax for different criteria).
Fill, Move, border, broadcast, center, colShape, composeGrowing, compress, compress2, compress3, connect, count, cut, embed, eraseCol, filterCol, getPos, getSize, getSize0, gravity, half, hull, hull0, insideMarked, interior, interior2, majCol, makeBorder, makeBorder2, mirror, myStack, pickMax, pickMaxes, pickNotMaxes, pickUnique, repeat, rigid, smear, splitAll, splitCols, splitColumns, splitRows, spreadCols, stackLine, toOrigin, wrap.

Each transformation takes either an image or a list of images, and produces either an image or a list of images. Image -&gt; image transformations are also broadcast across lists of images to produce list -&gt; list transformations. Transformations which take multiple inputs are given the original sample input image, or a black image with the dimensions of the output image, to make them unary.

Transformations are implemented to be O(#input pixels) to ensure no transformation takes significantly more time than any other, although some exceptions were made when they take negligible time for other reasons.
