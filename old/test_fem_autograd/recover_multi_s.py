"""Calculate the residual in the data feature solution."""

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from fealpy.mesh import TriangleMesh
from tqdm import trange
# import json
import time

from lafemeit.model import DataFeature, DataPreprocessor, MultiChannelFractional
from lafemeit.solver import LaplaceFEMSolver

torch.manual_seed(202400)
DEVICE = 'cuda:7' if torch.cuda.is_available() else 'cpu'
EXT = 63


def main(s0, noise, domain):
    ### Generate a phi as the target solution

    frac = MultiChannelFractional(252, len(s0), device=DEVICE)
    frac.from_npz(r"lafem/data/laplace_beltrami_63_63.npz")
    frac.initialize(s0)
    frac.requires_grad_(False)

    mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=EXT, ny=EXT, device=DEVICE)
    solver = LaplaceFEMSolver(mesh, p=1, reserve_mat=True)
    df_prepro = DataPreprocessor(solver)
    df_solver = DataFeature(solver, frac)
    gd = torch.from_numpy(np.load('lafem/data/cir3_e64_64_c8/gd/100.npy'))[0:len(s0)]
    gn = torch.from_numpy(np.load('lafem/data/cir3_e64_64_c8/gn.npy'))[0:len(s0)]

    gn = gn.broadcast_to(gd.shape)
    data = torch.stack([gd, gn], dim=-2).to(DEVICE)
    gnvn = df_prepro(data[None, ...])
    target_phi = df_solver(gnvn).detach().sum(dim=1, keepdim=False)
    print(f"Shape of target phi: {target_phi.shape}")

    ### Train a new s to see if converged

    # get the range of ipoints

    if domain == 'sub':
        ips = mesh.interpolation_points(p=1)
        flag1 = torch.logical_and(ips[:, 0] > -0.5, ips[:, 0] < 0.5)
        flag2 = torch.logical_and(ips[:, 1] > -0.5, ips[:, 1] < 0.5)
        flag = torch.logical_and(flag1, flag2).nonzero(as_tuple=True)[0]
    elif domain == 'full':
        flag = slice(None)
    else:
        raise ValueError(f"DOMAIN must be 'sub' or 'full', but got {domain}")

    frac.initialize([0.0] * len(s0))
    frac.requires_grad_(True)
    criterion = MSELoss()
    optim = Adam((frac.gamma,), lr=0.01, betas=(0.9, 0.98))

    # s_lists = []
    t1 = time.time()
    with torch.profiler.record_function("func_core"):

        for _ in trange(0, 1000):
            optim.zero_grad()
            noisy_gnvn = gnvn + torch.randn_like(gnvn) * gnvn * noise
            phi = df_solver(noisy_gnvn)
            phi = phi.sum(dim=1, keepdim=False)
            loss = criterion(phi[..., flag], target_phi[..., flag])
            loss.backward()
            optim.step()

    print(time.time() - t1)

            # s_lists.append(frac.gamma.detach().tolist())


    # data = {
    #     's0': s0,
    #     'noise': noise,
    #     'domain': domain,
    #     's_evo': list(zip(*s_lists)),
    # }

    # with open(f'test_fem_autograd/plot_data/multi_s_{time.time()}.json', 'w') as f:
    #     json.dump(data, f)


def run_profiling(s0, noise, domain, device):
    activities = []
    if device[:4] == 'cuda' and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    activities.append(torch.profiler.ProfilerActivity.CPU)  # 总是包含 CPU 活动

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=0, active=1),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=False,
        with_flops=True
    ) as prof:
        for _ in range(2):
            main(s0, noise, domain)
            prof.step()

    print(f"\n=== {device} Profiling Results ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total" if device == 'cuda' else "self_cpu_time_total",
        row_limit=10
    ))
    analyze_profiler(prof, device)


def analyze_profiler(prof, device):
    core_events = [event for event in prof.key_averages()]
    num_calls = len(core_events)
    if num_calls == 0:
        print(f"No {device} events recorded for func_core")
        return

    total_time = 0.0  # 单位：秒
    total_flops_profiled = 0

    for event in core_events:
        if device[:4] == 'cuda' and torch.cuda.is_available():
            # CUDA 时间（微秒）转换为秒
            time_us = event.cuda_time_total
            total_time += time_us / 1e6
            total_flops_profiled += event.flops
        else:
            # CPU 时间（微秒）转换为秒
            time_us = event.cpu_time_total
            total_time += time_us / 1e6

    flops_per_sec = total_flops_profiled / total_time
    print(f"FLOPS: {flops_per_sec / 1e9:.3f} GFLOPs (Profiled)")

    # if device == 'cuda' and torch.cuda.is_available():
    #     if total_time > 0:
    #         flops_per_sec = total_flops_profiled / total_time
    #         print(f"GPU FLOPS: {flops_per_sec / 1e12:.2f} TFLOPs (Profiled)")
    # else:
    #     # 手动计算 CPU FLOPs（假设每次调用为 2e9 FLOPs）
    #     total_flops_manual = num_calls * 2e9
    #     if total_time > 0:
    #         flops_per_sec = total_flops_manual / total_time
    #         print(f"CPU FLOPS: {flops_per_sec / 1e9:.2f} GFLOPs (Manual)")


if __name__ == "__main__":

    # s0 = [0.993, 0.314, 0.538, 0.934, 0.514, 0.760, 0.631, 0.931]
    s0 = [0.993, 0.314, 0.538, 0.934, 0.514]
    # s0 = [0.993, ]

    for noise in [0.2,]:
        run_profiling(s0, noise, "sub", DEVICE)
