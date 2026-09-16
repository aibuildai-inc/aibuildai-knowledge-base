# Code for model

Competition: seizure-detection
Rank: #27
Source: https://www.kaggle.com/c/seizure-detection/discussion/10092

<p>I've just released the code for my submission on <a href="https://github.com/streety/kaggle-seizure-prediction">github</a>, with discussion in a <a href="http://jonathanstreet.com/blog/seizure-detection-scikit-learn-pipelines/">blog post</a>.</p>

<p>I've cleaned the code up somewhat (added doc strings, removed commented out code) but it is largely what I had yesterday.</p>

<p>Plenty of room for improvement. For example, currently rather than reading each file once to import all the data it imports just one channel. To get all the channels it cycles through all the files as many times as there are channels.</p>

<p>Has anyone else released their code? What is the one thing you would do differently if you had to start again?</p>
