# Silver medal solution: ViT-L/14 with test time augmentations

Competition: google-universal-image-embedding
Rank: #40
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359371

Thank you to the host, organizers and all participants!

You can check out my solution here: https://www.kaggle.com/code/slavabarkov/google-universal-image-embedding-submission

Brief summary of my solution:
- ViT-L/14 encoder from the OpenCLIP implementation pre-trained on LAION-400M dataset, achieves 0.519 public/0.537 private if the input image is resized while maintaining the aspect ratio
- Projection layer trained with ArcFace on the Products-10k dataset based on @motono0223 work, achieves 0.598 public/0.608 private
- Multiple test time augmentations, achieves 0.628 public/0.631 private: 
	- Predefined crops - center crop and two crops with preselected offset
	- Pixel dropout - drops pixels from image according to predefined mask
