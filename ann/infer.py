
import numpy as np
import torch
import torch.nn.functional as F
import sucrose
from matplotlib import pyplot as plt

from lafemeit.utils import NPYDataset, NPZDataset

from train import AnnEIT, ccw_index


LOSS_FUNC = {
    "mse": F.mse_loss,
    "bce": F.binary_cross_entropy_with_logits
}


def main(case: str, data_id: int):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    ssc = sucrose.scenario("ann", case)

    model = ssc.partial(AnnEIT, "model")().to(device)
    loss_func = LOSS_FUNC[ssc["train.loss_func"]]
    ccw = ccw_index(252, device)
    print("Number of parameters:")
    NP = sum(p.numel() for p in model.parameters())
    print(NP/1000000, "M")

    ssc.load_state_dict(
        model = model,
        loader_kwds={"map_location": device}
    )
    model.eval()

    validate_data_dataset = NPYDataset(
        ssc["data.gd_folder"],
        names=[str(i) for i in range(0, 12000)],
    )

    validate_label_dataset = NPZDataset(
        ssc["data.inclusion_folder"],
        names = [str(i) for i in range(0, 12000)],
        channel_keys = []
    )

    gd = validate_data_dataset[data_id].to(dtype=torch.float32, device=device)
    gd = gd[..., ccw]
    gd = torch.roll(gd, 1, -1) - gd
    y_out = model(gd.reshape(1, -1)).reshape(64, 64) # (N, Nx, Ny)
    y_out = torch.clip(y_out, 0.0, 1.0)
    label = validate_label_dataset[data_id][0].reshape(y_out.shape)
    loss = loss_func(y_out, label.to(dtype=torch.float32, device=device))

    X = np.linspace(-1, 1, 64)
    XX, YY = np.meshgrid(X, X, indexing="ij")

    fig = plt.figure(figsize=(12, 6))
    axes = fig.add_subplot(121)
    axes.set_title(f"Loss: {loss.item():.3e}")
    ai = axes.pcolormesh(XX, YY, y_out.detach().cpu().numpy(), cmap="jet")
    fig.colorbar(ai)

    axes = fig.add_subplot(122)
    ai = axes.pcolormesh(XX, YY, label, cmap="jet")

    fig.tight_layout()
    fig.savefig(f"ann/figures/infer_{case.replace('/', '_')}_{data_id}.png")


if __name__ == "__main__":
    main("base3", 10001)