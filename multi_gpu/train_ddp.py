import os
import time
import json
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class SyntheticFightDataset(Dataset):
    def __init__(self, samples=200000, input_dim=1024, classes=6):
        g = torch.Generator().manual_seed(42)
        self.x = torch.randn(samples, input_dim, generator=g)
        self.y = torch.randint(0, classes, (samples,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class FightPolicyNet(nn.Module):
    def __init__(self, input_dim=1024, classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, classes),
        )

    def forward(self, x):
        return self.net(x)


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")

    dataset = SyntheticFightDataset(samples=args.samples)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = FightPolicyNet().to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Warmup
    model.train()
    x, y = next(iter(loader))
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()

    total_local_samples = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_local_samples += x.size(0)

    torch.cuda.synchronize()
    dist.barrier()

    elapsed = time.perf_counter() - start

    samples_tensor = torch.tensor(
        [total_local_samples],
        dtype=torch.long,
        device=device,
    )

    dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
    global_samples = int(samples_tensor.item())

    throughput = global_samples / elapsed

    if rank == 0:
        result = {
            "world_size": world_size,
            "gpu_model": torch.cuda.get_device_name(0),
            "epochs": args.epochs,
            "samples_per_epoch": args.samples,
            "global_batch_size": args.batch_size * world_size,
            "elapsed_seconds": elapsed,
            "processed_samples": global_samples,
            "throughput_samples_per_second": throughput,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "backend": "NCCL",
        }

        print()
        print("=" * 65)
        print("MULTI-GPU DDP BENCHMARK")
        print("=" * 65)
        print(f"GPUs:                    {world_size}")
        print(f"GPU model:               {result['gpu_model']}")
        print(f"Backend:                 NCCL")
        print(f"Global batch size:       {result['global_batch_size']}")
        print(f"Processed samples:       {global_samples}")
        print(f"Elapsed seconds:         {elapsed:.2f}")
        print(f"Throughput:              {throughput:,.0f} samples/sec")
        print("=" * 65)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

            print(f"Saved results to {args.output}")

    cleanup()


if __name__ == "__main__":
    main()
