
from tqdm import tqdm
from time import sleep

for i in tqdm(range(10), desc='outer'):
    for j in tqdm(range(10), desc='inner', leave=False):
        sleep(0.05)
