"""MNIST partitioned with PFLlib's official Dirichlet(alpha=0.1) label-skew scheme
(dataset/utils/dataset_utils.py, pulled verbatim from github.com/TsingZ0/PFLlib), fixed to
match this project's grok-family convention: only the real 60k MNIST train set is
partitioned across clients (Dirichlet skew); every client gets the full, untouched, real
10k MNIST test set (not a resplit of the partitioned pool). This project has no
personalization layer -- aggregation produces one shared global model -- so the only
meaningful generalization test is the true target distribution, same for every client,
regardless of how skewed each client's training partition is. The original
generate_mnist_dirichlet.py pooled train+test together and resplit 75/25 per client,
which measures generalization to each client's own local resample instead -- wrong
question for this project.
"""
import os
import sys
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, save_file

random.seed(1)
np.random.seed(1)


def generate_dataset(dir_path, num_clients, niid, balance, partition, n_train_total=60000):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    train_image = trainset.data.cpu().detach().numpy()
    train_label = trainset.targets.cpu().detach().numpy()
    test_image = testset.data.cpu().detach().numpy()
    test_label = testset.targets.cpu().detach().numpy()

    # Subsample the pool before partitioning (same fixed-seed-0 convention as
    # generate_mnist_grok_pathological.py) so a smaller n_train_total is a strict subset,
    # not a re-drawn sample -- keeps fraction sweeps nested/comparable.
    if n_train_total < len(train_image):
        pool_rng = np.random.default_rng(0)
        pool_idx = pool_rng.permutation(len(train_image))[:n_train_total]
        train_image = train_image[pool_idx]
        train_label = train_label[pool_idx]

    num_classes = len(set(train_label))
    print(f"Number of classes: {num_classes}, pool size: {len(train_label)}")

    # Partition only the (possibly subsampled) real train pool -- test stays the real,
    # untouched, full 10k set.
    X, y, statistic = separate_data((train_image, train_label), num_clients, num_classes,
                                     niid, balance, partition, class_per_client=2)

    train_data = [{"x": X[i], "y": y[i]} for i in range(num_clients)]
    test_data = [{"x": test_image, "y": test_label} for _ in range(num_clients)]

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)


if __name__ == "__main__":
    num_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_train_total = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    suffix = "-fulltest" if n_train_total == 60000 else f"-fulltest-N{n_train_total}"
    dir_path = f"MNIST-partition_dir-clients_{num_clients}-niid_True-balance_False-alpha_01{suffix}/"
    generate_dataset(dir_path, num_clients, niid=True, balance=False, partition="dir",
                      n_train_total=n_train_total)
