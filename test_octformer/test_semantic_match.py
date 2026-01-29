from local_grid import LocalGrid
import numpy as np
import torch
from time import time
from semantic_grid_ransac import Feature2DGlobalRegistrationPipeline

ref_cloud = np.load('clouds/ref_cloud.npz')['arr_0']
ref_probs = np.load('clouds/ref_probs.npz')['arr_0']
cand_cloud = np.load('clouds/cand_cloud.npz')['arr_0']
cand_probs = np.load('clouds/cand_probs.npz')['arr_0']

ref_grid = LocalGrid(semantic_probability_threshold=0.3, floor_height=-0.9, ceil_height=1.5)
ref_grid.update_from_cloud_and_transform(ref_cloud[:, :3], ref_probs)
cand_grid = LocalGrid(semantic_probability_threshold=0.3, floor_height=-0.9, ceil_height=1.5)
cand_grid.update_from_cloud_and_transform(cand_cloud[:, :3], cand_probs)

pipeline_semantic = Feature2DGlobalRegistrationPipeline(outlier_thresholds=[2.5, 1.0, 0.5, 0.25, 0.25],
                                                        ransac_iterations=1000)
ref_grid_tensor = torch.Tensor(ref_grid.layers['occupancy'])
cand_grid_tensor = torch.Tensor(cand_grid.layers['occupancy'])
ref_grid_semantic = ref_grid.layers['semantic']
ref_grid_semantic_tensor = torch.Tensor(ref_grid_semantic)
cand_grid_semantic = cand_grid.layers['semantic']
cand_grid_semantic_tensor = torch.Tensor(cand_grid_semantic)
t1 = time()
transform_semantic, score_semantic = pipeline_semantic.infer(ref_grid_tensor, ref_grid_semantic,
                                  cand_grid_tensor, cand_grid_semantic,
                                  verbose=True)
t2 = time()
print('Matching time:', t2 - t1)
print('Fitness:', score_semantic)