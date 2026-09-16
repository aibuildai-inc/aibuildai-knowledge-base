# 6th place solution

Competition: mayo-clinic-strip-ai
Rank: #6
Source: https://www.kaggle.com/c/mayo-clinic-strip-ai/discussion/359562

Hi, this competition is my first challenge at kaggle. I really enjoyed this competition and thought hard for about three months. I don't know why I could obtain a gold medal, but I'm sure it's thanks to everyone who have shared the notebooks or suggestions. 
Here I describe my solution in brief.

Submission kernel:
https://www.kaggle.com/code/kodaihatayama/mayo02-submission

### Preprocessing
I created tiles with the following steps. **I didn't resize the image** in all preprocessing steps.
- Cut 6 rectangles of width 512 at equal intervals from every image 
(The purpose of this step is to create tiles quickly and avoid memory crashes without resizing.)
- Make tiles of size 512x512 from the above rectangles and pick up the top 8 dark tiles per image
- Split each tile to instances of size 32x32 for training

For using pyvips, I refered to the following kernel. Thank you, @analokamus !
https://www.kaggle.com/code/analokamus/how-to-use-pyvips-offline

### Validation and Undersampling
I used the hold-out method with undersampling. 
- Split patient id to train and validation (80:20) which was stratified by class (CE, LAA)
- Perform undersampling to be CE:LAA = 1:1 on the train dataset only

### Model
I created the ensemble model of the following 2 methods.
  - CNN 
  - lightGBM

The difference of private LB between CNN only model and the ensemble model was a bit (approximately 0.676 → 0.668).
As a side note, I created a model per institution (center id). (This approach may not be effective against test dataset because the institutions of test may be different from them of train.)

For CNN, I refered to the following kernel. Thank you, @vbookshelf !
https://www.kaggle.com/code/vbookshelf/cnn-how-to-use-160-000-images-without-crashing

### What didn't work for me
- grayscale
- histogram equalization

Thanks for reading.
