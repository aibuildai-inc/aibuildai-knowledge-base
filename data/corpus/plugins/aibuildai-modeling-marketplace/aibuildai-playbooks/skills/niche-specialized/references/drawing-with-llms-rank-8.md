# 8th Place Solution

Competition: drawing-with-llms
Rank: #8
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/581095

Congratulations to all the winners, and thank you kaggle for hosting this very interesting and challenging competition. Thank you @philippsinger for being a great teammate as always.

## Local Validation
We generated a few hundred prompt / questions samples for local validation. While scores on LB were usually a bit lower, we saw good correlation with that setup but high variance on both, local score and on public LB score. In the end, this led us to pick two high scoring subs with basically only different seeds to mitigate a bit of randomness.

## The Solution Pipeline
Our pipeline consists of a SDXL model, that we fine tuned the best images from 16 previous runs of prompted Flux Schnell and SDXL models and based on the score on one of our local validation sets. We made most use of the infrastructure by utilizing both GPUs in parallel to generate images and to score in a single thread for each. The kaggle package made this a bit hard to set up. And we had to go with an API based approach to prevent context errors. Apparently, this is now fixed on newer kagglehub versions, so it shouldn't be an issue for upcoming competitions using the package approach.

With that optimized pipeline, we were able to generate and iterate through about 14-18 images per sample, based on the compute requirements of the rest of the pipeline. The bitmaps were subsequently converted to SVG using optimized VTracer settings and some minification steps, mostly leveraging the `scour` lib and reducing the original image size that it fits the 10kB requirement. For OCR, we added a small "+" in one of the corners, which was very robust and cheap, but judging by other competitors not ideal in terms of aesthetic score.

Finally, all images were scored with a proxy prompt for vqa and only applying MedianBlur to reduce runtime. 

```
qa = {
    'question': [f'Does this look like {prompt}?'], 
    'choices': [["yes", "no"]], 
    'answer': ["yes"]
}
```

The best image was picked and returned for each sample.

## Other Things we Tried
We did invest quite some time into diffvg and differentiable image preprocessing steps. While we had some promising results in easier pipelines and also locally, ultimately, "Package Prediction Error" stopped us from using it in our final submission.
We also invested time into hacking the aesthetic score using fully differentiable preprocessing steps, but only had very limited success with all the defense mechanics in place and while keeping VQA score high. 

Apparently, we were missing diffvg and using the similarity scores between prompt and image for a better final rank. Nevertheless, we are very grateful for this challenge that always sparked new ideas to try out. I am honestly surprised by the great quality of the final images.
