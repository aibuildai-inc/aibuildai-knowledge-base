# [31st Place] Object Detection approach

Competition: shopee-product-matching
Rank: #31
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238322

Hi there!

First of all, I would like to congratulate all of you for the incredible talent that was shown during this competition. Moreover, I would like to thank my teammates ([Miguel](https://www.kaggle.com/miguelbm), [Pablo](https://www.kaggle.com/talavantecodes) and [Carlos](https://www.kaggle.com/carlosbort)) for all the things they have taught me along the way and that resulted in a **silver medal** in my **first ever Kaggle competition**! Finally, I would like to personally congratulate my teammate [Alejandro Lanaspa](https://www.kaggle.com/alelafe) for reaching the *Kaggle Master* tier after this silver medal! Having mentioned that, let's dig into the main purpose of this topic:

I created this topic to share one of the approaches that my team used during this competition and that I was in charge of implementing: Object Detection. As its name suggests, this approach was based on detecting objects in the image of each product, with the objective to have another item of comparison between different elements.

From the technical point of view, YOLOv5m was used as the model. The library where this model is implemented can be found [here](https://github.com/ultralytics/yolov5); and an effort was made to use the pretrained model offline so it was available during submissions (see [the dataset](https://www.kaggle.com/atmguille/yolov5-git) that I created for that purpose). In terms of the information that this model provides, its output is not only a list of objects. In fact, the following data is produced:
1. Names of the objects detected.
2. Confidence for each detection.
3. Coordinates of the bounding box where each object is detected.

The example code that generates this data for the Shopee train set can be found in [this notebook](https://www.kaggle.com/atmguille/shopee-object-detection-generate-data) (list format) and [this notebook](https://www.kaggle.com/atmguille/shopee-object-detection-generate-data-one-hot) (one-hot format). I also uploaded the output of these notebooks to [this dataset](https://www.kaggle.com/atmguille/shopee-train-with-objects) so any of you could directly use it without executing anything.

Now, let’s analyze what further details were obtained from these items:
1. This is the obvious output, but thanks to it we could filter objects that were relevant to the competition. By default, the model tries to detect 80 different types of objects, which are listed [here](https://www.kaggle.com/atmguille/shopee-train-with-objects?select=objects_names.txt). In this list, one can find very different objects from each other, from ‘baseball bat’ to ‘elephant’, that are clearly unrelated to what should be detected in this case. After observing the results and computing different statistics (number of different elements with an object, number of different groups, percentage of completeness of a group with an object… See [this notebook](https://www.kaggle.com/atmguille/shopee-object-detection-analysis) for the complete analysis) it was determined that two of the most relevant ones were ‘person’, in images of people advertising clothes, and ‘bottle’, in cosmetics and food items.
2. Although at the beginning we thought of including objects that the model was very sure of, we detected that it was missing some important objects that were detected with a lower confidence level. Also, so as to have as many options as possible to select from in the algorithm that is described at the end of the following point, we were interested in setting a low confidence level that obviously outputs more detected objects.
3. The following information was obtained from the coordinates:
    - By having these coordinates, what we really have are the four vertices of a rectangle. Thanks to this, one can estimate the size of the detected object by computing the area of this rectangle. After that, bigger objects could be taken into account with higher priority.
    - Another factor to consider when assigning importance to the different objects in an image is how centered they are. In fact, the products that are advertised (the ones we are more interested in) are normally located in the middle of the image. Based on this, it would be interesting to compute how far an object is from being at the middle of the picture. Can that be done with the available information? Sure! In a more graphical way, the value that we are looking for is the length of the red line shown below: 
    - Once we have an idea of how big objects are and how centered they are, apart from using this information to prioritize some objects over others; it could be used to remove background noise, focusing just on the advertised product. Based on this idea, an approach that we took was to crop the most important detected object from the image. To select this important object, the biggest and the most centric object were obtained, and a simple algorithm selected the best option considering how centric was the biggest object and how big was the most centric one, with prevalence for the most centric. To get an idea on how useful this crop could be, see the following examples obtained from the Shopee dataset where the bounding boxes are colored in red. Which equality should be easier to check, the first or the second one? 
 
The answer is clear, isn’t it? If you are not convinced yet, see the following example with totally different backgrounds: 
*Note*: the cropped images were not obtained by hand but by the algorithm explained before. The code that actually does these crops can be found in [this notebook](https://www.kaggle.com/atmguille/shopee-object-detection-crop-images)




Hope you enjoyed this post and found my high-level description interesting. I will be very happy to answer any question or comment anything you would like. Furthermore, I would love to also read any feedback that you could have, since as I mentioned above this is my first experience with Kaggle and I am willing to improve! 😄
