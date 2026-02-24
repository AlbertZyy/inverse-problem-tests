
import sys
sys.path.append("./pinn")


from model import PINNEIT, MixedLoss, ccw_index
from scheduler import MyLR_Scheduler
from dataset_new import NPZDataset, MemoryDataset

import torch
from torch.optim import Adam
from torch.utils.data import RandomSampler, BatchSampler
import sucrose
from tqdm import tqdm


def main(case: str):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    ssc = sucrose.scenario("pinn", case)

    model = ssc.partial(PINNEIT, "model")().to(device)
    optim = ssc.partial(Adam, "optim")(model.parameters())
    scheduler = ssc.partial(MyLR_Scheduler, "scheduler")(optim)
    loss_func = MixedLoss().to(device)
    ccw = ccw_index(252, device)
    print("Number of parameters:")
    NP = sum(p.numel() for p in model.parameters())
    print(NP/1000000, "M")

    ssc.load_state_dict(
        model = model,
        loss_func = loss_func,
        optim = optim,
        scheduler = scheduler,
        loader_kwds={"map_location": device}
    )

    train_data_dataset = NPZDataset(
        ssc["data.gd_folder"],
        names=[str(i) for i in range(ssc['data.train_set_start'], ssc['data.train_set_end'])],
    )
    train_data_dataset = MemoryDataset(
        train_data_dataset.names_seq,
        train_data_dataset._read_data,
        num_workers=4,
        device=device,
        tqdm=True
    )

    validate_data_dataset = NPZDataset(
        ssc["data.gd_folder"],
        names=[str(i) for i in range(ssc['data.valid_set_start'], ssc['data.valid_set_end'])],
    )
    validate_data_dataset = MemoryDataset(
        validate_data_dataset.names_seq,
        validate_data_dataset._read_data,
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
            data = train_data_dataset[indices]
            delta_V = data["delta_V"].to(dtype=torch.float32) # (N, 8, 252)
            delta_V = delta_V[:, 0:1, ccw] # use only the first frequency, (N, 1, 252)
            delta_V.requires_grad_(True)
            optim.zero_grad()
            sigma_pred, bdry_V_pred, all_V_pred = model(delta_V)
            # (N, 4096), (N, 1, 252), (N, 1, 4096)
            sigma_train, bdry_V, full_V = data["sigma"], data["bdry_V"][:, 0:1, :], data["full_V"][:, 0:1, :]
            # sigma_train = sigma_train.reshape(sigma_pred.shape)
            # bdry_V = bdry_V.reshape(bdry_V_pred.shape)
            # full_V = full_V.reshape(all_V_pred.shape)
            loss = loss_func(delta_V, sigma_pred, sigma_train, bdry_V_pred, bdry_V, all_V_pred, full_V)
            loss.backward()
            optim.step()
            scheduler.step()
            ssc.step()

        writer.add_scalar('loss(train)', loss.item(), ssc.num_steps)
        ssc.save_state_dict(
            10000, model=model, loss_func=loss_func, optim=optim, scheduler=scheduler
        )
        model.eval()

        sampler = RandomSampler(validate_data_dataset)
        batch_sampler = BatchSampler(sampler, batch_size=ssc["train.batch_size"], drop_last=False)
        losses = []

        for indices in tqdm(batch_sampler,
                        desc=f'Epoch {epoch + 1}/{ssc["train.epochs"]}', ascii=True,
                        unit='batch', leave=False, position=1):
            delta_V = validate_data_dataset[indices]["delta_V"].to(dtype=torch.float32)
            delta_V = delta_V[:, 0:1, ccw] # use only the first frequency, (N, 1, 252)
            delta_V.requires_grad_(True)
            sigma_pred, bdry_V_pred, all_V_pred = model(delta_V)
            sigma_train, bdry_V, full_V = validate_data_dataset[indices]["sigma"], validate_data_dataset[indices]["bdry_V"][:, 0:1, :], validate_data_dataset[indices]["full_V"][:, 0:1, :]
            sigma_train = sigma_train.reshape(sigma_pred.shape)
            bdry_V = bdry_V.reshape(bdry_V_pred.shape)
            full_V = full_V.reshape(all_V_pred.shape)
            loss = loss_func(delta_V, sigma_pred, sigma_train, bdry_V_pred, bdry_V, all_V_pred, full_V)
            losses.append(loss.item())

            loss_mean = sum(losses) / len(losses)

        writer.add_scalar('loss(validate)', loss_mean, ssc.num_steps)

    writer.close()


if __name__ == '__main__':
    main("base")
