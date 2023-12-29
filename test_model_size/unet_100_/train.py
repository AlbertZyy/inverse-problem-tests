
import sys

sys.path.append('./src')

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from unet_100 import build_model
from dataset import NPZDataset

SAVE = True
GPU_ID = 3
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')


def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    val = - torch.log(y_pred) * y_true - torch.log(1 - y_pred) * (1 - y_true)
    return torch.mean(val)

model, MODEL_NAME = build_model(device, s=0.0, s_grad=True)

optim = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-9)
train_dataset = NPZDataset("./data/gdgn_64_64_train/", 2000)
validate_dataset = NPZDataset("./data/gdgn_64_64_validate/", 100)
loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
loader_2 = DataLoader(validate_dataset, batch_size=20, shuffle=True)

writer_1 = SummaryWriter(f'./test_model_size/unet_100_/log_{MODEL_NAME}', flush_secs=30)


def train(epoch: int):
    step = 0

    for gdgn, label in loader:
        optim.zero_grad()

        y_out = model(gdgn.to(device=device)) # (N, 1, Nx, Ny)
        loss = loss_fn(y_out, label.flatten().to(dtype=torch.float32, device=device))
        loss.backward()
        optim.step()
        step += 1
        print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")
        writer_1.add_scalar('loss(train)', loss.item(), epoch*100 + step)
        writer_1.add_scalar('s', model.df_solver._frac.s.item(), epoch*100 + step)

    if SAVE:
        torch.save(model.state_dict(), f"./test_model_size/unet_100_/{MODEL_NAME}.pth")


def validate(epoch):
    gdgn, label = next(iter(loader_2))
    with torch.no_grad():
        y_out = model(gdgn.to(device=device))
        loss = loss_fn(y_out, label.flatten().to(dtype=torch.float32, device=device))
        print(f"epoch: {epoch}, loss: {loss.item()}")
    writer_1.add_scalar('loss(validate)', loss.item(), epoch*100+100)


for epoch in range(10):
    train(epoch)
    validate(epoch)

writer_1.close()
