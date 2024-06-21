
import sys

sys.path.append('src')

import numpy as np
import torch
from torch.utils.data import RandomSampler, BatchSampler
from fealpy.torch.mesh import TriangleMesh

from fem import EITDataPreprocessor, DataFeatureFEMSolver, LaplaceFEMSolver
from fdm import LaplaceFDMSolver
from data_feature import MultiChannelDataFeature
from dataset import NPYDataset, NPZDataset


fem_dataset = NPYDataset('data/cir3_e64_64_c8_test/gd', names=[str(i) for i in range(50)])
fem_gn = torch.from_numpy(np.load('data/cir3_e64_64_c8_test/gn.npy'))

fdm_dataset = NPZDataset('data/cir3_e64_64_c8_old_test',
                         names=[str(i) for i in range(50)],
                         channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'])


mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=63, ny=63)
fem_solver = LaplaceFEMSolver(mesh, p=1, q=3)
fem_dfprepro = EITDataPreprocessor(fem_solver)
fem_dfsolver = DataFeatureFEMSolver(fem_solver)

fdm_solver = LaplaceFDMSolver([63, 63], [2/63, 2/63])
fdm_dfsolver = MultiChannelDataFeature(fdm_solver)


INEDX = 8
