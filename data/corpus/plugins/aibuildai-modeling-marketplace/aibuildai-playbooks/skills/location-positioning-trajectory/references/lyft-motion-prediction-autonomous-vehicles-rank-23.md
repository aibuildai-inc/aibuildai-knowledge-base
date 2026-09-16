# 23rd solution (single model based on Resnet18)

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #23
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/200035

This was a nice dataset to play with. It is like HPC simulations, one have to relax and let things run for long. To my surprise, one can go very far using simple concepts and published work. My solution is simply an implementation of the two papers from Uber
1- Uncertainty-aware Short-term Motion Prediction of Traffic Actors for Autonomous Driving, https://arxiv.org/abs/1808.05819
2- Multimodal Trajectory Predictions for Autonomous Driving using Deep Convolutional Networks, https://arxiv.org/abs/1809.10732

Now to some details which I think contributed to my best performing model:

1- Sample the data to break the serial correlation in time. I only trained on frames with 60+10+1 gaps. I wanted to avoid data leakages so I kept a gap of history length (10) + future prediction (50) + (1) extra. Also, if a frame is too crowded (i.e. too many agents), I sampled a sub-set of the agents (20 in my code below).

```
def get_dataset_masks(
        zarr_path,
        zarr_dt,
        th_agent_prob: float,
        min_frame_history: int,
        min_frame_future: int,
        scene_mask=None,
        chop_data=False,
        chop_idx_list=[100, 200],
        chop_agents=False):

    """

    Modified from create_chopped_dataset


    Returns:
        mask: numpy array of bool to pass to AgentDatasetExtended
    """

    if scene_mask is None:
        scene_mask = np.ones(len(zarr_dt.scenes), dtype=np.bool)
    else:
        assert len(zarr_dt.scenes) == len(scene_mask), "mask should be equal length"

    agents_mask_path = Path(zarr_path) / f"agents_mask/{th_agent_prob}"

    if not agents_mask_path.exists():  # don't check in root but check for the path
        assert 0
    agents_mask_original = np.asarray(convenience.load(str(agents_mask_path)))

    agents_mask = np.zeros(len(zarr_dt.agents), dtype=np.bool)
    # for scene in zarr_dt.scenes.get_mask_selection(scene_mask):
    for scene_idx in range(len(zarr_dt.scenes)):
        if scene_mask[scene_idx] == 0:
            continue
        scene = zarr_dt.scenes[scene_idx]
        if chop_data:
            for num_frame_to_copy in chop_idx_list:
                kept_frame = zarr_dt.frames[scene["frame_index_interval"][0] + num_frame_to_copy - 1]
                agents_slice = get_agents_slice_from_frames(kept_frame)
                # In create_chopped_dataset: no mask for min_frame_history
                mask = agents_mask_original[agents_slice][:, 1] >= min_frame_future
                num_agents_per_frame = mask.sum()
                max_agents_per_frame = 20
                if chop_agents and num_agents_per_frame > max_agents_per_frame:  # or 10
                    removed_indices = np.random.choice(
                        np.where(mask)[0],
                        num_agents_per_frame - max_agents_per_frame,
                        replace=False)
                    mask[removed_indices] = False
                agents_mask[agents_slice] = mask.copy()

        else:
            first_frame = zarr_dt.frames[scene["frame_index_interval"][0]]
            last_frame = zarr_dt.frames[scene["frame_index_interval"][1] - 1]
            agents_slice = get_agents_slice_from_frames(first_frame, last_frame)

            past_mask = agents_mask_original[agents_slice][:, 0] >= min_frame_history
            future_mask = agents_mask_original[agents_slice][:, 1] >= min_frame_future
            mask = past_mask * future_mask
            agents_mask[agents_slice] = mask.copy()

    return agents_mask
```
2-  I had a clean implementation of the Multiple-Trajectory Prediction (MTP) loss. I trained on the MoN loss and validated on the nll loss

```
def uber_like_loss_new(gt, pred, confidences, avails):
    batch_size, num_modes, future_len, num_coords = pred.shape

    # ensure that your model outputs logits
    gt = gt[:, None, :, :]  # add modes
    avails = avails[:, None, :, None]  # add modes and cords
    l2_error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability
    l2_error = torch.sum(l2_error, dim=-1)  # reduce future_len

    best_mode_target = torch.argmin(l2_error, dim=1).detach()
    classification_loss = torch.nn.functional.cross_entropy(confidences, best_mode_target, reduction='none')

    alpha = 1.0
    MoN_error = classification_loss + alpha * l2_error[torch.arange(batch_size), best_mode_target]
    MoN_error = MoN_error.reshape(-1, 1)
    error = torch.nn.functional.log_softmax(confidences, dim=1) - 0.5 * l2_error  # reduce future_len
    max_value, _ = torch.max(error, dim=-1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    nll_error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value
    return MoN_error, nll_error
```

3- I extracted some meta data from the AgentDataset. Here is a minimal implementation 

```
class AgentDatasetExtended(AgentDataset):
    def __init__(
        self,
        cfg,
        zarr_dataset,
        rasterizer,
        perturbation,
        agents_mask,
        min_frame_history,
        min_frame_future,
        transform,
        l5kit_version,
    ):
        assert perturbation is None, "AgentDataset does not support perturbation (yet)"
        super(AgentDatasetExtended, self).__init__(
            cfg, zarr_dataset, rasterizer, perturbation, agents_mask, min_frame_history, min_frame_future)
        self.min_frame_future = min_frame_future
        self.min_frame_history = min_frame_history
        self.transform = transform
        self.l5kit_version = l5kit_version

    def __getitem__(self, index: int) -> dict:
        """
        Differs from parent returning the indices of the frame, agent and scene
        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        index = self.agents_indices[index]
        track_id = self.dataset.agents[index]["track_id"]
        frame_index = bisect.bisect_right(self.cumulative_sizes_agents, index)
        scene_index = bisect.bisect_right(self.cumulative_sizes, frame_index)

        if scene_index == 0:
            state_index = frame_index
        else:
            state_index = frame_index - self.cumulative_sizes[scene_index - 1]
        data_dic = self.get_frame(scene_index, state_index, track_id=track_id)

        # track_id = self.dataset.agents[index]["track_id"]
        # centroid = self.dataset.agents[index]["centroid"]
        # yaw = self.dataset.agents[index]["yaw"]
        velocity = self.dataset.agents[index]["velocity"]
        label_probabilities = self.dataset.agents[index]["label_probabilities"]
        # data_dic['track_id2'] = np.int64(track_id)
        # data_dic['centroid2'] = centroid
        # data_dic['yaw2'] = yaw
        data_dic['velocity'] = velocity
        data_dic['label_probabilities'] = label_probabilities

        data_dic['ego_translation'] = self.dataset.frames[frame_index]["ego_translation"]
        data_dic['ego_rotation'] = self.dataset.frames[frame_index]["ego_rotation"]  # matrix
        data_dic['timestamp'] = self.dataset.frames[frame_index]["timestamp"]
        data_dic['hour'] = datetime.fromtimestamp(data_dic['timestamp'] / 1e9).hour
        data_dic['weekday'] = datetime.fromtimestamp(data_dic['timestamp'] / 1e9).weekday()

        if self.transform:
            data_dic = self.transform(data_dic)
        return data_dic
```

4- I had a second head of 47 inputs (probably too much), which is fed all sort of metadata (label_probabilities, yaw, extent, acceleration, ego_translation, ego_centroid_diff, hour, weekday) and then passed through two fully connected layers before merging with the pooling layers of the backbone. This concatenated vector is then passed to dense layer, relu and then the output layer.

5- I applied a `cumsum` function to NN output to force the NN to produce something like the differences, then I applied the image_to_world_matrix transformation before calling the loss function of the real coordinate. I wanted the output of my NN model to be in the image space (conceptually). 

There are many small additional details. I used lookahead optimizer wrapped around ADAM with starting learning rate of 1e-4 that and then decreased the LR by 0.99 every 100000 steps over batches of 32 samples (which takes about an hour on my machine RTX-Titan-X). Learning rate is reduced after 20 iterations of the 320k samples. Best solution is reached after 110 of these 320k samples (best model Private LB 13.580, Public LB 14.265). I then averaged 7 models based on checkpointing (Private LB 13.154, Public LB 13.722). My image size is 336x336 with a resolution of 0.25x0.25, trained on full data with the following frames `chop_idx_list_train = [11, 11 + 61, 11 + 2 * 61, 11 + 3 * 61]`

Probably, few simple modification could produce a bit better score.
(a) A bigger patch of 64 samples might perform better (diversity of the samples). 
(b) Change the backbone to ResNet50 or MobileNet-V2
