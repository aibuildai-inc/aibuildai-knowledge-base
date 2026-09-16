# 15th Place Solution: SD3.5M + Vtracer + select best from 3 image

Competition: drawing-with-llms
Rank: #15
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/581051

Thanks to the Kaggle and participants of the competition. Special thanks to @richolson, who shared his ideas and notebooks, and to many other participants whose every word provided food for thought.

<h2>Solution</h2>
As for my solution, it was SD3.5M with prompt_suffix = "Flat illustration, stereotypical". Plus vtracer with binary search for optimal parameters. Plus paligemma2-10b-mix-448 to select the best of three images.

The successful submission only came near the end of the competition at night. The main boost came from splitting the image description into three questions.

<h2>Main boost</h2>

For short descriptions, the split was similar to TF-IDF:

description: 'a purple forest at dusk'
question: ['Is purple forest?', 'Is forest at?', 'Is at dusk?']

For longer descriptions, the text was simply split into three parts:

description: 'purple pyramids spiraling around a bronze cone'
question: ['Is purple pyramids?', 'Is spiraling around?', 'Is bronze cone?']
