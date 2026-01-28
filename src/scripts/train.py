"""
Train sparse autoencoders on activations from a diffusion model.
"""

import os
import sys
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from simple_parsing import parse

from src.sae.config import TrainConfig
from src.sae.trainer import SaeTrainer


@dataclass
class RunConfig(TrainConfig):
    mixed_precision: str = "no"

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    seed: int = 42
    """Random seed for shuffling the dataset."""
    device: str = "cuda"
    num_epochs: int = 1


def load_dataset_from_npy_batch_dir(dir, dtype=torch.float32):
    """
    Creates a Hugging Face Dataset from a folder of .npy files.

    Args:
        dir (str): Path to the folder containing .npy files.
        dtype: Data type for the tensors (default: torch.float32)

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    data_dir = Path(dir)
    timestep = int(data_dir.parent.name[1:])
    file_list = sorted(data_dir.glob('*.npy'))

    if not file_list:
        raise ValueError(f"No files found in {dir}")

    def gen():
        for file_path in file_list:
            try:
                batch = np.load(file_path, mmap_mode='r')
                for sample in batch:
                    sample = sample / np.linalg.norm(sample, ord=2)
                    yield {
                        "activations": sample[np.newaxis, ...],
                        "timestep": timestep
                    }

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    dataset = Dataset.from_generator(gen)
    dataset.set_format(
        type="torch",
        columns=["activations", "timestep"],
        dtype=dtype,
    )

    return dataset


def load_dataset_from_npy_dir(dir, dtype=torch.float32):
    data_dir = Path(dir)
    timestep = int(data_dir.name[1:])
    file_paths = sorted(data_dir.glob('*.npy'))

    if not file_paths:
        print(f"Found no .npy files in {data_dir}")
        exit()

    def data_generator():
        for f_path in file_paths:
            sample_data = np.load(f_path)
            yield {
                "activations": sample_data,
                "timestep": timestep
            }

    dataset = Dataset.from_generator(data_generator)

    # Set format for each dataset
    dataset.set_format(
        type="torch",
        columns=["activations", "timestep"],
        dtype=dtype,
    )
    return dataset


def load_datasets_from_npy_dirs(base_dirs, hookpoint, dtype=torch.float32):
    """
    Load and concatenate datasets from multiple directories each containing samples as .npy files.

    Args:
        base_dirs (list[str]): List of base directory paths containing the datasets
        hookpoint (str): Name of the hookpoint directory
        dtype: Data type for the tensors (default: torch.float32)

    Returns:
        Dataset: Concatenated dataset
    """
    datasets = []
    print(f"Concatenating datasets from {base_dirs}")

    for base_dir in base_dirs:
        # dataset = load_dataset_from_npy_dir(base_dir, dtype)
        dataset = load_dataset_from_npy_batch_dir(base_dir, dtype)
        datasets.append(dataset)

    # Concatenate all datasets
    return concatenate_datasets(datasets)


def load_datasets_from_dirs(base_dirs, hookpoint, dtype=torch.float32):
    """
    Load and concatenate datasets from multiple directories.

    Args:
        base_dirs (list[str]): List of base directory paths containing the datasets
        hookpoint (str): Name of the hookpoint directory
        dtype: Data type for the tensors (default: torch.float32)

    Returns:
        Dataset: Concatenated dataset
    """
    datasets = []
    print(f"Concatenating datasets from {base_dirs}")

    for base_dir in base_dirs:
        dataset = Dataset.load_from_disk(
            os.path.join(base_dir, hookpoint), keep_in_memory=False
        )

        # Set format for each dataset
        dataset.set_format(
            type="torch",
            columns=["activations", "timestep"],
            dtype=dtype,
        )

        datasets.append(dataset)

    # Concatenate all datasets
    return concatenate_datasets(datasets)


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)
    # add output_or_diff to the run name
    args.run_name = args.run_name + f"_{args.dataset_path[0].split('/')[-2]}"

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    args.dtype = dtype
    print(f"Training in {dtype=}")
    # Awkward hack to prevent other ranks from duplicating data preprocessing
    dataset_dict = {}
    if not ddp or rank == 0:
        for hookpoint in args.hookpoints:
            if len(args.dataset_path) > 1:
                try:
                    dataset = load_datasets_from_dirs(args.dataset_path, hookpoint, dtype)
                except Exception:
                    dataset = load_datasets_from_npy_dirs(args.dataset_path, hookpoint, dtype)
            else:
                try:
                    dataset = Dataset.load_from_disk(
                        os.path.join(args.dataset_path[0], hookpoint), keep_in_memory=False
                    )
                except Exception:
                    # dataset = load_dataset_from_npy_dir(args.dataset_path[0], dtype=dtype)
                    dataset = load_dataset_from_npy_batch_dir(args.dataset_path[0], dtype=dtype)
            dataset.set_format(
                type="torch",
                columns=["activations", "timestep"],
                dtype=dtype,
            )
            dataset = dataset.shuffle(args.seed)
            if limit := args.max_examples:
                dataset = dataset.select(range(limit))
            dataset_dict[hookpoint] = dataset
            print(f"Loaded dataset for {hookpoint}")
    # NOTE: DDP not tested so far
    if ddp:
        dist.barrier()
        if rank != 0:
            for hookpoint in args.hookpoints:
                dataset = Dataset.load_from_disk(
                    os.path.join(args.dataset_path, hookpoint), keep_in_memory=False
                )
                dataset.set_format(
                    type="torch",
                    columns=["activations", "timestep"],
                    dtype=dtype,
                )
                dataset = dataset.shuffle(args.seed)
                dataset = dataset.shard(dist.get_world_size(), rank)
                dataset_dict[hookpoint] = dataset
                print(f"Loaded dataset for {hookpoint}")

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        trainer = SaeTrainer(args, dataset_dict)

        trainer.fit()


if __name__ == "__main__":
    run()
