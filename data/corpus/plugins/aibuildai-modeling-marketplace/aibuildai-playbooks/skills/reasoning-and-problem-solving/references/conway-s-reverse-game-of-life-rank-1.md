# Congratulations Miranda!

Competition: conway-s-reverse-game-of-life
Rank: #1
Source: https://www.kaggle.com/c/conway-s-reverse-game-of-life/discussion/7254

<p>Congratulations to Miranda, as well as Martin O'Leary and Cromarty Rockall! I thought I was using a strong approach of brute force calculation (cluster-based) and an algorithm that exploited the symmetry of the problem, but I couldn't catch any of you!</p>
<p>I am curious to know approaches and hardware (single core/multi-core/SSE/AVX/GPU/FPGA/other) people used. I used OpenMP and MPI to run on an older (E5420 Intel Xeon 2.5GHz) cluster with 376 cores. I ran out of time to implement a SIMD (SSE-based) algorithm. This problem seems well matched to a GPU-based implementation (which I did not do).</p>
<p>Algorithmically, I computed conditional probabilities of a starting cell_ij being dead/alive based on a 7x7 block of stopping cells centered at i &amp; j. I folded in the pi/2 rotational symmetry and the inversion symmetry (are there other symmetries that I missed?).</p>
<p>Regards,</p>
<p>Jason</p>
