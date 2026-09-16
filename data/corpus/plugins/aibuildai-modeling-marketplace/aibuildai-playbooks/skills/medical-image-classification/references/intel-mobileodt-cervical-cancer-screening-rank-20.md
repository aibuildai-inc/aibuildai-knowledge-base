# 20th place solution/approach

Competition: intel-mobileodt-cervical-cancer-screening
Rank: #20
Source: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/35168

My strategy was as per below (I quickly verified it in the beginning by doing manual crops and training a simple network which gave 0.73 on LB):

Annotated 3 points on external OS (two extremes and mid point), so the idea was to get the location of external OS and approximate zoom of the external OS. Took max distance between these 3 points as a indicator of size of external OS. Based on this size indicator, I zoomed out/in images to get uniform sized external OS images and cropped them around external OS center. For this I used custom CNN and MSE as loss function, but could not get sufficiently low loss.  Overall network performed well in giving external OS and surrounding areas. 

Next stage was to feed cropped images got from above to a CNN for classification. I used network architectures similar to inception, resnet and xception with reduced filters as these networks were over-fitting the data. I used just one fold CV on train images and another fold on train+additional images(manually removed some duplicates). Trained the networks on these two sets of data and took average of approximate 10 models prediction for final submission.

Looking at the other solutions, I have a feeling that I should have worked with more folds, specially on the train image data. I still have a feeling that producing better crops from stage-1 should have given better results. I did not retrain model based on stage 1 ground truths for final submission.

Overall it was a great learning experience for me. Thanks Kaggle and sponsors for providing this opportunity.
