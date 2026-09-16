# 4th Place Solution - Split images into tiles and do 3D CNN

Competition: mayo-clinic-strip-ai
Rank: #4
Source: https://www.kaggle.com/c/mayo-clinic-strip-ai/discussion/364466

Maybe It's too late to write a solution now, but I want to share my ideas.
Because it's a good thing to share good ideas in Kaggle.
also I would like to thank the organizers for hosting the great competition.  

# Summary  
Important features of the competition data are that the image sizes given to us are very large, with an average of approximately 40000x40000, having many white background.  
The solution I came up with was to remove background area of images and to split the images into tiles and train a 3D CNN Network.  
My method showed a very stable Local CV without any post-processing.  
  
# Details





