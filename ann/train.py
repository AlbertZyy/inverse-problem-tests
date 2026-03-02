
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import RandomSampler, BatchSampler
from lafemeit.utils import NPYDataset, NPZDataset, MemoryDataset
import sucrose
from tqdm import tqdm


class AnnEIT(nn.Module):
    ACTIVATE_MAP = {
        "relu": torch.relu,
        "tanh": torch.tanh,
        "softplus": F.softplus,
        "leaky_relu": F.leaky_relu
    }
    def __init__(self, input_dims: int, hidden_dims: list[int], output_dims: int, activation: str = "relu"):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dims, elementwise_affine=False)
        self.input_layer = nn.Linear(input_dims, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dims)
        self.activate = AnnEIT.ACTIVATE_MAP[activation]

    def forward(self, x: Tensor):
        x = self.layer_norm(x)
        x = self.activate(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activate(layer(x))

        return self.output_layer(x)


def ccw_index(edge_length: int, device=None):
    NN = edge_length
    assert NN % 4 == 0
    EM = NN // 4
    original = torch.arange(NN, dtype=torch.int32, device=device)
    bottom_edge = original[EM+1:3*EM:2]
    right_edge = original[3*EM:4*EM]
    top_edge = torch.flip(original[EM:3*EM-1:2], dims=[0])
    left_edge = torch.flip(original[0:EM], dims=[0])
    return torch.concat([bottom_edge, right_edge, top_edge, left_edge], dim=0)

LOSS_FUNC = {
    "mse": F.mse_loss,
    "bce": F.binary_cross_entropy_with_logits
}


def main(case: str):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    ssc = sucrose.scenario("ann", case)

    model = ssc.partial(AnnEIT, "model")().to(device)
    optim = ssc.partial(SGD, "optim")(model.parameters())
    loss_func = LOSS_FUNC[ssc["train.loss_func"]]
    ccw = ccw_index(252, device)
    print("Number of parameters:")
    NP = sum(p.numel() for p in model.parameters())
    print(NP/1000000, "M")

    ssc.load_state_dict(
        model = model,
        optim = optim,
        loader_kwds={"map_location": device}
    )

    train_data_dataset = NPYDataset(
        ssc["data.gd_folder"],
        names=[str(i) for i in range(ssc['data.train_set_start'], ssc['data.train_set_end'])],
    )
    train_data_dataset = MemoryDataset(
        train_data_dataset.names,
        train_data_dataset.read_data,
        num_workers=4,
        device=device,
        tqdm=True
    )

    train_label_dataset = NPZDataset(
        ssc["data.inclusion_folder"],
        names = [str(i) for i in range(ssc['data.train_set_start'], ssc['data.train_set_end'])],
        channel_keys = []
    )
    train_label_dataset = MemoryDataset(
        train_label_dataset.names_seq,
        train_label_dataset._read_data,
        num_workers=4,
        device=device,
        tqdm=True
    )

    validate_data_dataset = NPYDataset(
        ssc["data.gd_folder"],
        names=[str(i) for i in range(ssc['data.valid_set_start'], ssc['data.valid_set_end'])],
    )
    validate_data_dataset = MemoryDataset(
        validate_data_dataset.names,
        validate_data_dataset.read_data,
        num_workers=4,
        device=device,
        tqdm=True
    )

    validate_label_dataset = NPZDataset(
        ssc["data.inclusion_folder"],
        names = [str(i) for i in range(ssc['data.valid_set_start'], ssc['data.valid_set_end'])],
        channel_keys = []
    )
    validate_label_dataset = MemoryDataset(
        validate_label_dataset.names_seq,
        validate_label_dataset._read_data,
        num_workers=4,
        device=device,
        tqdm=True
    )

    writer = ssc.start_pytorch_tensorboard()

    for epoch in tqdm(ssc.epoch_range(ssc["train.epochs"]), ascii=True, unit='epoch', leave=False, position=0):
        model.train()
        sampler = RandomSampler(train_data_dataset)
        batch_sampler = BatchSampler(sampler, batch_size=ssc['train.batch_size'], drop_last=False)

        for indices in tqdm(batch_sampler,
                            desc=f'  Epoch {epoch + 1}/{ssc["train.epochs"]}', ascii=True,
                            unit='batch', leave=False, position=1):
            N = len(indices)
            gd = train_data_dataset[indices].to(dtype=torch.float32) # (N, 8, 252)
            gd = gd[..., ccw]
            gd = torch.roll(gd, 1, -1) - gd
            optim.zero_grad()
            y_out = model(gd.reshape(N, -1)).squeeze(1) # (N, 1, Nx, Ny)
            label = train_label_dataset[indices].reshape(y_out.shape)
            loss = loss_func(y_out, label.to(dtype=torch.float32))
            loss.backward()
            optim.step()
            ssc.step()

        writer.add_scalar('loss(train)', loss.item(), ssc.num_steps)
        ssc.save_state_dict(10000, model=model, optim=optim)
        model.eval()

        sampler = RandomSampler(validate_data_dataset)
        batch_sampler = BatchSampler(sampler, batch_size=ssc["train.batch_size"], drop_last=False)
        losses = []

        with torch.no_grad():
            for indices in tqdm(batch_sampler,
                            desc=f'Epoch {epoch + 1}/{ssc["train.epochs"]}', ascii=True,
                            unit='batch', leave=False, position=1):
                N = len(indices)
                gd = validate_data_dataset[indices].to(dtype=torch.float32)
                gd = gd[..., ccw]
                gd = torch.roll(gd, 1, -1) - gd
                y_out = model(gd.reshape(N, -1)).squeeze(1) # (N, 1, Nx, Ny)
                label = validate_label_dataset[indices].reshape(y_out.shape)
                loss = loss_func(y_out, label.to(dtype=torch.float32))
                losses.append(loss.item())

            loss_mean = sum(losses) / len(losses)

        writer.add_scalar('loss(validate)', loss_mean, ssc.num_steps)

    writer.close()


if __name__ == '__main__':
    main("base3")
    # print(ccw_index(252, None))
