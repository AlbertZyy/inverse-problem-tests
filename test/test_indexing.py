
import sys

sys.path.append('./src')

from fdm import Indexing, UniformPartition
from dataset import NPZDataset

dataset = NPZDataset('./data/gdgn_64_64_train', 10)
gdgn, labels = dataset[0]
gd = gdgn[:, 0, :]
gn = gdgn[:, 1, :]
print(gd.shape)

EXT = 63
H = 2./EXT

indexing = UniformPartition([EXT, EXT], [H, H])
source = indexing.neumann_source(gn.T)

print(source.shape)
print(source)
