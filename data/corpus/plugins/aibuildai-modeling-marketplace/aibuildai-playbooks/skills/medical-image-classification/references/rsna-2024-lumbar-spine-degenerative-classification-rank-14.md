# 14th place solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #14
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539459

### Models 
- Single-stage multi-condition (s12): 0.412 CV
- Multi-stage multi-condition (s1+s2): 0.404 CV
- Multi-stage single-condition (@tamotamo shared it [here](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539459#3012444)): 0.400 CV

### Data
CLAHE normalization of the input.  I projected input points of interest provided by orgs for specific sources and can locate all 25 points in each of the inputs. I used [publicly shared key point annotation](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/528653) for a two-stage setup. For external data, I used [Spider ](https://huggingface.co/datasets/cdoswald/SPIDER)(dataset for segmentation of specific vertebra and disks) in my very first experiments and single-stage model pre-training.

### Single-stage model (s12)
**Architecture**:  Attention-based aggregation (similar to SED used in Bird-call competition): I predict 3d voxel grid (at reduced res corresponding to 1/token size) of attention weights and predicted targets. Then I apply spatial softmax on attention and weight the targets accordingly summing them along the spatial dimensions. It results in scalar Bx25 predictions. I use [competition metric approximation](https://www.kaggle.com/code/junkoda/optimize-the-evaluation-metric) as the loss function. In addition, I consider aux loss on attention using Gaussians to approximate annotated key points.
As an alternative, I consider 2 cross-attention layer decoder applied to each slice independently. The decoder takes 25 learnable queries as the input and each predicts the target, xy position, and probability p that this slice is valid for a particular target. The final prediction is weighted (based on p) target and xy. The loss is CE on p, Competition metric approximation on the target, and MSE on xy. The idea here is that the decoder trying to predict the position of the specific key point is also aggregating the information providing the target class. This approach performs similarly to attention-based aggregation.

**Backbone**: Since the model must be able to predict specific levels for visually indistinguishable vertebra it should have access to the global image content (i.e. ViT-based architecture is preferable), and in early experiments with Spider I saw that DINOv2 (with registers) can assign levels to vertebra reliably. Therefore, I used DINOv2 B backbone taking a sequence of input frames (N,3,H,W). For side views the backbone is augmented with **zero initialized LSTM adapters** to perform sequence mixing and is pre-trained on the Spider segmentation dataset. For axial views, I added a **LSTM mixing layer** after the backbone.

The image setup: 16x448x448 for side input and 48x332x332 for axial input. If the sequence is shorter, images are repeated.

### 2-stage multi-condition (s1+s2)
s1 is a simple single-slice segmentation model applied to the center slice and trained to predict 10 key points shared publicly. I used Sagittal T1 + Sagittal T2/STIR sources and then reprojected corresponding coordinates to find appropriate slices in Axial T2 and also cross-check the sources and drop points in the side view. I do not do s1 on axial view because multiple groups make the data messy and the task is quite harder than point detection in a lateral view. The backbone is DINOv2 again because I need to predict 10 distinguishable key points. With 10 instead of 5 key poins I can derive the pox orientation and the size. Input size 448x448.

Then I apply s2 for each source. In the side view I crop a small stack of boxes around proper areas of interest. As a result, I produce 5 stacks of image crops 5x16x3x192x192 for side views. Axial cropping is done based on the plane corresponding to the projected key point and is not so aggressive laterally. So I produce sequences of 5x8x3x320x320 (making sure the group=view angle is the same for all selected images, and replicate images if the number of selected slices is insufficient). The model is simple ConvNeXtv2 nano + LSTM mixing layer + concat pooling over sequence + the head for side view, and in axial view I use DINOv2 instead. The competition metric approximation is used as the loss function.

### Aggregation
I use a simple weighting of each model independent for each condition (apply to logits). The matrix below shows the contribution of each of 7 sources (s2 Sagittal T2/STIR, s2 Sagittal T1, s2 Axial T2, s12  Sagittal T2/STIR, s2 Sagittal T1, s2 Axial, Multi-stage single-condition) to 5 targets (SCS, L NFN, R NFN, L SS, RSS). Some sources are particularly important for specific targets. 
w =    [[0.1228, 0.0025, 0.0031, 0.0612, 0.1021],
        [0.0549, 0.2573, 0.1362, 0.0484, 0.1198],
        [0.1712, 0.0790, 0.0269, 0.2366, 0.1519],
        [0.1308, 0.0068, 0.0373, 0.1285, 0.0949],
        [0.0161, 0.1436, 0.2533, 0.0101, 0.0072],
        [0.0689, 0.0220, 0.0208, 0.0797, 0.1492],
        [0.4353, 0.4888, 0.5225, 0.4354, 0.3749]]
CV 0.379, LB 0.35/0.41
