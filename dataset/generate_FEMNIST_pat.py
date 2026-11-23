"""
Pathological (label-skew) partition of the full FEMNIST pool.

Pools every image from every writer in LEAF's images_by_writer.pkl (writer identity is ignored),
then gives each of the 62 classes to exactly `clients_per_class` distinct clients.
With 20 clients and 4 clients per class there are 62*4 = 248 class slots, so 8 clients hold
13 classes and 12 clients hold 12. A class's samples are divided among its clients with the same
unbalanced scheme as utils.dataset_utils.separate_data (balance=False), then each client is
split 75/25 into train/test with split_data.

Prerequisite: LEAF FEMNIST preprocessing, with LEAF's 'data' folder renamed to FEMNIST/
(see generate_FEMNIST.py).

Usage (from this directory):
    python generate_FEMNIST_pat.py [num_clients] [clients_per_class] [balance|unbalance]
    python generate_FEMNIST_pat.py 20 4 unbalance
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool
from PIL import Image
import torchvision.transforms as transforms
from generate_FEMNIST import relabel
from utils.dataset_utils import check, split_data, save_file


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(ROOT_PATH, 'FEMNIST', 'intermediate', 'images_by_writer.pkl')
NUM_CLASSES = 62

# raw NIST PNGs are 128x128; resize to 28x28 like LEAF, 1 channel, scaled to [-1, 1]
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


def load_image(rel_path):
    # LEAF stores paths as 'data/raw_data/...'; the LEAF 'data' folder was renamed to 'FEMNIST'
    path = os.path.join(ROOT_PATH, 'FEMNIST', rel_path.replace('data/', '', 1))
    return transform(Image.open(path).convert("L")).numpy()


def assign_classes(num_clients, clients_per_class):
    """Each class goes to `clients_per_class` distinct clients; clients get as equal a class count as possible."""
    total_slots = NUM_CLASSES * clients_per_class
    capacity = np.full(num_clients, total_slots // num_clients)
    capacity[np.random.choice(num_clients, total_slots % num_clients, replace=False)] += 1

    # greedy: each class goes to the clients with the most free slots (random tie-break), so class sets
    # overlap but are not identical across clients; owner order is shuffled so the remainder share of
    # an unbalanced split does not always land on the same client
    class_to_clients = {}
    for k in np.random.permutation(NUM_CLASSES):
        order = np.lexsort((np.random.rand(num_clients), -capacity))
        owners = order[:clients_per_class]
        assert np.all(capacity[owners] > 0)
        capacity[owners] -= 1
        class_to_clients[int(k)] = np.random.permutation(owners).tolist()
    assert np.all(capacity == 0)
    return class_to_clients


def generate_dataset(save_path, num_clients, clients_per_class, balance):
    config_path = os.path.join(save_path, 'config.json')
    train_path = os.path.join(save_path, 'train') + '/'
    test_path = os.path.join(save_path, 'test') + '/'

    if check(config_path, train_path, test_path, num_clients, True, balance, 'pat'):
        return

    data = pd.read_pickle(META_PATH)
    paths = [p for _, imgs in data for p, _ in imgs]
    labels = np.array([relabel(l) for _, imgs in data for _, l in imgs])
    print(f"Pool: {len(data)} writers, {len(paths)} images")

    print("Loading images ...")
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        images = np.stack(pool.map(load_image, paths, chunksize=2000)).astype(np.float32)  # (N, 1, 28, 28)

    class_to_clients = assign_classes(num_clients, clients_per_class)
    dataidx_map = {c: [] for c in range(num_clients)}
    for k, owners in class_to_clients.items():
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        num_per = len(idx_k) / len(owners)
        if balance:
            num_samples = [int(num_per) for _ in range(len(owners) - 1)]
        else:
            num_samples = np.random.randint(max(num_per / 10, 1), num_per, len(owners) - 1).tolist()
        num_samples.append(len(idx_k) - sum(num_samples))
        start = 0
        for client, n in zip(owners, num_samples):
            dataidx_map[client].extend(idx_k[start:start + n].tolist())
            start += n

    X, y, statistic = [], [], []
    for client in range(num_clients):
        idxs = np.array(dataidx_map[client])
        X.append(images[idxs])
        y.append(labels[idxs])
        statistic.append([(int(k), int(np.sum(y[client] == k))) for k in np.unique(y[client])])
        print(f"Client {client}\t Size of data: {len(idxs)}\t Num classes: {len(statistic[client])}\t Labels: ", np.unique(y[client]))
    del images

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, NUM_CLASSES,
              statistic, True, balance, 'pat')


if __name__ == "__main__":
    num_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    clients_per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    balance = len(sys.argv) > 3 and sys.argv[3] == "balance"

    random.seed(1)
    np.random.seed(1)

    save_path = os.path.join(ROOT_PATH, f"FEMNIST-partition_pat-clients_{num_clients}-cpc_{clients_per_class}"
                                        f"-balance_{balance}-full-train_075")
    generate_dataset(save_path, num_clients, clients_per_class, balance)
