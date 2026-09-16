# Silver medal solution: ViT, ConvNext, kNN-regression with BLIP-2 captions

Competition: stable-diffusion-image-to-prompts
Rank: #33
Source: https://www.kaggle.com/c/stable-diffusion-image-to-prompts/discussion/410610

Thank you to the host, organizers, all the participants and my teammate @manwithaflower!

Brief summary of our solution:
- Models trained to predict prompt embeddings (ViT, ConvNext):
   - First, train a separate MLP projection head for each model using selected image-text pairs from the LAION dataset. Filter the images based on image/text cosine similarity, retaining only the highest-scoring ones. This actually results in a dataset primarily consisting of images of written text with related prompts containing that same text - but it is still surprisingly effective, achieving 0.54434 public/0.54351 private on a single model.
   - Fine-tune the models further with selected prompt/image pairs. Filter the prompts based on cosine similarity, the more aggressive filtering the better, with our final filtering threshold of 0.55 cosine similarity. Mainly use data from DiffusionDB 14M, @xiaozhouwang GPT2 & Hardcoded datasets, @jeinsong ChatGPT dataset, @motono0223 dataset, images from Discord downloaded with URLs from Open Prompts dataset, and images downloaded from URLs in @tanreinama 900k dataset.
   - Apply layerwise learning rate decay for fine-tuning.
   - Use a loss function based on angular distance, which outperforms raw cosine similarity.
   - For consistent local validation, filter prompts from the training dataset that are similar to validation prompts based on cosine similarity.
   - Determine ensemble weights for the models using Nelder-Mead minimization algorithm on local validation.
   The resulting ensemble achieves 0.59333 public/0.59109 private on the leaderboard.
</br>
- Retrieval-based/kNN-regression approach with BLIP-2 image captioning:
   - Improve @motono0223's kNN-regression approach:
     - Filter out duplicate text embeddings from the datasets.
     - Combine multiple sub-arrays of numpy arrays.
     - Penalize distances between sample and examples differently by simple distance exponentiation.
     - Add more diverse data.
     - Memory-map stored reference prompt embedding arrays for efficient access without loading the files into memory.

        This allows to reduce inference time to 1 hour and achieves 0.57179 public/0.56952 private.
   - Generate 20 diverse image captions for each test image using BLIP-2 and diverse beam search decoding. Store these embeddings along with the corresponding reference embeddings, and use them with the retrieval-based/kNN-regression approach. This achieves 0.58821 public/0.58617 private.

- Ensemble both approaches! The resulting ensemble scores 0.61004 public/0.60787 private on the leaderboard.
