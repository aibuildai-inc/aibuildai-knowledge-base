# Go with the flow (3rd place)

Competition: sartorius-cell-instance-segmentation
Rank: #3
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/297984

*This post describes part of the 3rd place solution that's based on **cellpose***

# Backstory
Going into this competition I had no experience with image segmentation. I knew a library called Detectron exists so I started with reading tutorials and learning to use that and it gave me good early LB results. Taking the pretrained model from the LiveCell repo and finetuning it on the competition data was good enough to put me in the top 10 for a while during the first month. Meanwhile I've seen many people trying U-net in public notebooks and on the forum but all the shared results were far bellow mask-rcnn scores so I discarded it as an inferior approach. That is until start of December when I stumbled upon a mention of [cellpose](http://www.cellpose.org/) on the forum - the example images they show on their site looked promissing and the paper claims *"significantly outperforming Stardist and Mask R-CNN"* so I've decided to try to run it myself.

Imagine my excitement when the very first model I trained scored 0.307 on the LB. No training code written, no postprocessing, just saved annotations into png files and run the train script. Comparable out of the box detectron model, without any tuning, was scoring around 0.28 so I started to realize that U-net might not be an inferior architecture after all. I battled with myself whether to share the finding. On the one hand I found it very cool, but I also felt I could be giving a powerful tool into hands of my competitors with only few weeks go. In the end I shared the initial experiment and then conciously refrained from any more comments or mentions of it :)

From there I followed the same game plan that worked with mask rcnn
1. pretrain on livecell
2. tune postprocessing on different cell types
3. ensemble mutliple models
4. add unlabled data annotated by a larger ensamble (inconclusive)

With all that I managed to push it into the public score with single model: 0.333, and best 5 models ensemble: 0.335. This didn't end up as the strongest model in our team submissions (@alexandrecc mask r-cnn claimed that) but I think it provided helpful diversity in the ensemble.

# How it works
The key idea is that rather than train U-net directly on the binary masks it first creates an intermediate representation of "flow mask". To generate that first it calculates the center of mass of each mask. And then all the other points represent an angle to flow from that point towards the center. This makes it easier to disentangle touching or overlapping masks - which is the main issue a regular U-Net faces.


The model itself is fairly simple, 20 conv layers on both down and up paths. Total of 6 million paramters which is less than resnet18. After the training the model output on the same image looks like this:

You can see many of the centers identified correctly and how it's able to separate even tightly packed cells. (this image still only scores 0.155 showing how difficult a high astro score is)

From this output they follow the flows from each pixel to it's center to go back to binary masks. I must admit I don't understand the exact details of this. The paper says *"the predicted flow fields are used to construct a dynamical system with fixed points whose basins of attraction represent the predicted masks."*

#The gory details
What I said earlier about how easy it was to get a baseline solution unfortunately didn't translate into ease of further refinement. I have spent a lot of time on experimenting and digging through their code in the last four weeks.

I hit the first wall when trying to train it on LiveCell. The library insists on loading all the images and masks into the memory before the training even starts -this is not a good idea when working with ten thousand of 520x704 images. I ended up pulling the relevant bits out of the library and writing a fast.ai dataloader and training script.

Even though the model itself is small the training utilzes heavy augmentations and high zooms on small areas of an image. So you need to train for a lot of epochs to fully cover all the data. (I was using 300-500 epochs). When I started training LiveCell with my new dataloaders I've noticed that now the disk IO is the bottleneck - spinning at 100% while GPU waits idly. I had to rearange stuff on my system to free enough SSD space and only then I got a smooth training fully utilizing the GPU.

A key detail in cellpose training is that it tries to maintain the pixel size of cells within a fixed range. At train time it's fairly simple - just look at the ground truth sizes and resize the image accordingly. At inference that's harder as you need to guess what the cell sizes are. That's the dreaded **diameter** discussed in a couple of threads here on the forum. Setting this correctly has a huge impact on models performance and I've spent so much time fiddling with it. What I have tried:
- setting a constant value per cell type
- setting a range of values and use that as TTA, 
- using cellpose's "size models" - which are linear regressions run on the U-net innermost activations
- using output of a detectron model and taking sizes from there. 

In my final submission I have an ensamble of 10 models, use the first one with the "size model" and then pass the diamater to the following models. At each step recalculating it on the predicted masks to improve the estimate. This works ok, but I feel like this a weakness of this model that would require more exploring.
