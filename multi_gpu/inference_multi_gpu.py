import argparse
import json
import time

import torch
import torch.nn as nn
import torch.multiprocessing as mp


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


def worker(rank, batches, batch_size, input_dim, queue):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = FightPolicyNet(input_dim=input_dim).to(device)
    model.eval()

    x = torch.randn(batch_size, input_dim, device=device)

    with torch.inference_mode():
        for _ in range(20):
            _ = model(x)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(batches):
            _ = model(x)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    samples = batches * batch_size

    queue.put({
        "rank": rank,
        "samples": samples,
        "elapsed": elapsed,
        "throughput": samples / elapsed,
    })


def run(world_size, batches, batch_size, input_dim):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    processes = []

    for rank in range(world_size):
        p = ctx.Process(
            target=worker,
            args=(rank, batches, batch_size, input_dim, queue),
        )
        p.start()
        processes.append(p)

    results = [queue.get() for _ in range(world_size)]

    for p in processes:
        p.join()

    total_samples = sum(r["samples"] for r in results)
    elapsed = max(r["elapsed"] for r in results)

    return {
        "world_size": world_size,
        "gpu_model": torch.cuda.get_device_name(0),
        "batch_size_per_gpu": batch_size,
        "batches_per_gpu": batches,
        "processed_samples": total_samples,
        "elapsed_seconds": elapsed,
        "throughput_samples_per_second": total_samples / elapsed,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.gpus > torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {args.gpus} GPUs but only "
            f"{torch.cuda.device_count()} are available."
        )

    result = run(
        args.gpus,
        args.batches,
        args.batch_size,
        args.input_dim,
    )

    print()
    print("=" * 65)
    print("MULTI-GPU INFERENCE BENCHMARK")
    print("=" * 65)
    print(f"GPUs:                    {result["world_size"]}")
    print(f"GPU model:               {result["gpu_model"]}")
    print(f"Batch size / GPU:        {result["batch_size_per_gpu"]}")
    print(f"Processed samples:       {result["processed_samples"]}")
    print(f"Elapsed seconds:         {result["elapsed_seconds"]:.2f}")
    print(
        f"Throughput:              "
        f"{result["throughput_samples_per_second"]:,.0f} samples/sec"
    )
    print("=" * 65)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
