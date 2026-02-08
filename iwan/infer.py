
import numpy as np
import torch
import sucrose
from matplotlib import pyplot as plt

from train import DenseNet

sucrose.logger.setLevel("INFO")
DTYPE = torch.float32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONTEXT = {"dtype": DTYPE, "device": DEVICE}
I_CONTEXT = {"dtype": torch.int32, "device": DEVICE}


def main(case: str):
    GD = 2
    ssc = sucrose.scenario("iwan", case)

    u_net = ssc.partial(DenseNet, "u.")(input_dim=GD).to(DEVICE)
    gamma_net = ssc.partial(DenseNet, "gamma.")(input_dim=GD).to(DEVICE)

    bdry_info = np.load(ssc["data.bdry_info_path"])
    node = torch.from_numpy(bdry_info["node"]).to(**CONTEXT)
    label = np.load(ssc["data.label_path"])['label']
    label = label * 9 + 1.0

    ssc.load_state_dict(
        u_net = u_net,
        gamma_net = gamma_net,
        loader_kwds={"map_location": DEVICE}
    )
    u_net.eval()
    gamma_net.eval()

    gamma = gamma_net(node)
    grid = bdry_info["node"].reshape(64, 64, 2)

    fig = plt.figure(figsize=(12, 6))
    axes = fig.add_subplot(121)
    ai = axes.pcolormesh(grid[..., 0], grid[..., 1], gamma.detach().cpu().numpy().reshape(64, 64))
    fig.colorbar(ai)

    axes = fig.add_subplot(122)
    ai = axes.pcolormesh(grid[..., 0], grid[..., 1], label.reshape(64, 64))

    fig.savefig(f"iwan/figures/infer_{case.replace('/', '_')}.png")


if __name__ == "__main__":
    main("base")
