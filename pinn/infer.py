
import numpy as np
import torch
import torch.nn.functional as F
import sucrose
from matplotlib import pyplot as plt

from dataset_new import NPZDataset
from model import PINNEIT, ccw_index


LOSS_FUNC = {
    "mse": F.mse_loss,
    "bce": F.binary_cross_entropy_with_logits
}


def main(case: str, data_id: int):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    ssc = sucrose.scenario("pinn", case)

    model = ssc.partial(PINNEIT, "model")().to(device)
    loss_func = LOSS_FUNC["bce"]
    ccw = ccw_index(252, device)
    print("Number of parameters:")
    NP = sum(p.numel() for p in model.parameters())
    print(NP/1000000, "M")

    ssc.load_state_dict(
        model = model,
        loader_kwds={"map_location": device}
    )
    model.eval()

    validate_data_dataset = NPZDataset(
        ssc["data.gd_folder"],
        names=[str(i) for i in range(ssc['data.train_set_start'], ssc['data.valid_set_end'])],
    )

    data = validate_data_dataset[data_id]
    delta_V = data["delta_V"].to(dtype=torch.float32) # (8, 252)
    print("delta_V shape:", delta_V.shape)
    delta_V = delta_V[None, :, ccw] # (1, 8, 252)
    y_out, _, _ = model(delta_V) # (1, Nx, Ny)
    y_out = torch.clip(y_out.reshape(64, 64), 0.0, 1.0)
    label = data["sigma"].reshape(y_out.shape)
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
    fig.savefig(f"pinn/figures/infer_{case.replace('/', '_')}_{data_id}.png")


if __name__ == "__main__":
    main("base2", 10653)