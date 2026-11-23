"""Generate an IID MNIST partition for the grokking positive control (Omnigrok, arXiv
2210.01117): each client gets a disjoint random n_train_per_client-sample subset of the
60k MNIST training set (their headline demo uses n_train_per_client=1000), and the full
10k MNIST test set. Not present in this PFLlib checkout (no generate_mnist.py existed),
so written from scratch following the schema read_client_data expects:
dataset/<name>/{train,test}/<client_id>.npz with a 'data' key -> {"x": ..., "y": ...}.
"""
import argparse
import json
import os

import numpy as np
import torchvision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument("--n_train_per_client", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=str, default="../dataset")
    args = parser.parse_args()

    name = f"MNISTGrok_n{args.n_train_per_client}_c{args.num_clients}"
    base = os.path.join(args.root, name)
    train_dir = os.path.join(base, "train")
    test_dir = os.path.join(base, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    raw_dir = os.path.join(args.root, "mnist_raw")
    train_set = torchvision.datasets.MNIST(root=raw_dir, train=True, download=True)
    test_set = torchvision.datasets.MNIST(root=raw_dir, train=False, download=True)

    train_x = (train_set.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
    train_y = train_set.targets.numpy().astype(np.int64)
    test_x = (test_set.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
    test_y = test_set.targets.numpy().astype(np.int64)

    total_needed = args.num_clients * args.n_train_per_client
    if total_needed > len(train_x):
        raise ValueError(
            f"num_clients*n_train_per_client={total_needed} exceeds MNIST train size {len(train_x)}"
        )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_x))

    for cid in range(args.num_clients):
        idx = perm[cid * args.n_train_per_client:(cid + 1) * args.n_train_per_client]
        np.savez_compressed(
            os.path.join(train_dir, f"{cid}.npz"),
            data=np.array({"x": train_x[idx], "y": train_y[idx]}, dtype=object),
        )
        np.savez_compressed(
            os.path.join(test_dir, f"{cid}.npz"),
            data=np.array({"x": test_x, "y": test_y}, dtype=object),
        )

    config = {
        "num_clients": args.num_clients,
        "num_classes": 10,
        "n_train_per_client": args.n_train_per_client,
        "n_test_per_client": len(test_x),
        "iid": True,
        "seed": args.seed,
        "source": "Omnigrok arXiv:2210.01117 -- IID random subsample, shared full test set",
    }
    with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote {name}: {args.num_clients} clients x {args.n_train_per_client} train samples, "
          f"{len(test_x)} shared test samples")


if __name__ == "__main__":
    main()
