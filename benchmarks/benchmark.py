"""
Benchmark VPTree<784> (compiled from vptree.cpp via pybind11) against a
NumPy brute-force nearest-neighbor baseline on the MNIST-784 dataset.

Expects a CSV at ./mnist_784.csv with 785 columns (784 pixels + 1 label)
or 784 columns (pixels only). Header row optional.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import vptree as vptree_lib  # pure-Python reference implementation (PyPI: vptree)
import vptree_mnist


REPO = HERE
DEFAULT_CSV = REPO_ROOT / "data" / "mnist_784.csv"


def load_mnist(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MNIST CSV not found at {csv_path}. "
            "Download from Kaggle (aadeshkoirala/mnist-784) and place it there."
        )
    df = pd.read_csv(csv_path)
    if df.shape[1] == 785:
        # drop a label column named 'class' / 'label' / whatever is non-pixel
        pixel_cols = [c for c in df.columns if c not in ("class", "label", "target")]
        if len(pixel_cols) == 785:
            pixel_cols = df.columns[:-1]  # fallback: assume label is last
        X = df[pixel_cols[:784]].to_numpy(dtype=np.float64)
    elif df.shape[1] == 784:
        X = df.to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unexpected column count: {df.shape[1]}")
    return np.ascontiguousarray(X)


def brute_force_nn(data: np.ndarray, queries: np.ndarray) -> np.ndarray:
    # ||q - x||^2 = ||q||^2 - 2 q.x + ||x||^2 ; take sqrt of the min per row
    data_sq = np.einsum("ij,ij->i", data, data)
    dists = []
    chunk = 256
    for start in range(0, queries.shape[0], chunk):
        q = queries[start:start + chunk]
        q_sq = np.einsum("ij,ij->i", q, q)
        d2 = q_sq[:, None] - 2.0 * q @ data.T + data_sq[None, :]
        d2 = np.maximum(d2, 0.0)  # numerical safety
        dists.append(np.sqrt(d2.min(axis=1)))
    return np.concatenate(dists)


def run(csv: Path, sizes: list[int], n_queries: int, seed: int,
        max_lib_n: int) -> pd.DataFrame:
    print(f"Loading {csv} ...")
    X_all = load_mnist(csv)
    print(f"Loaded MNIST: shape={X_all.shape}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(X_all.shape[0])
    X_all = X_all[idx]

    # queries are drawn from a held-out slice so we're not just self-querying
    max_n = max(sizes)
    if X_all.shape[0] < max_n + n_queries:
        raise ValueError("Dataset too small for requested N + queries")

    queries = X_all[max_n:max_n + n_queries].copy()

    rows = []
    for n in sizes:
        data = np.ascontiguousarray(X_all[:n])
        print(f"\n[N={n}]")

        t0 = time.perf_counter()
        tree = vptree_mnist.VPTree(data)
        t_vp_build = time.perf_counter() - t0
        print(f"  vp build:         {t_vp_build:8.3f} s")

        t0 = time.perf_counter()
        kd = cKDTree(data)
        t_kd_build = time.perf_counter() - t0
        print(f"  kd build:         {t_kd_build:8.3f} s")

        # Pure-Python reference VPTree. Slow: gated by --max-lib-n.
        run_lib = n <= max_lib_n
        if run_lib:
            dist_fn = lambda a, b: float(np.linalg.norm(a - b))
            data_list = list(data)
            t0 = time.perf_counter()
            lib_tree = vptree_lib.VPTree(data_list, dist_fn)
            t_lib_build = time.perf_counter() - t0
            print(f"  lib build:        {t_lib_build:8.3f} s")
        else:
            t_lib_build = float("nan")
            print(f"  lib build:        skipped (N > {max_lib_n})")

        t0 = time.perf_counter()
        vp_dists = tree.nearest_batch(queries)
        t_vp = time.perf_counter() - t0
        print(f"  vp query (x{n_queries}):    {t_vp:8.3f} s  ({t_vp / n_queries * 1e3:.3f} ms/query)")

        t0 = time.perf_counter()
        kd_dists, _ = kd.query(queries, k=1)
        t_kd = time.perf_counter() - t0
        print(f"  kd query (x{n_queries}):    {t_kd:8.3f} s  ({t_kd / n_queries * 1e3:.3f} ms/query)")

        if run_lib:
            t0 = time.perf_counter()
            lib_dists = np.array([lib_tree.get_nearest_neighbor(q)[0] for q in queries])
            t_lib = time.perf_counter() - t0
            print(f"  lib query (x{n_queries}):   {t_lib:8.3f} s  ({t_lib / n_queries * 1e3:.3f} ms/query)")
        else:
            t_lib = float("nan")
            lib_dists = None

        t0 = time.perf_counter()
        bf_dists = brute_force_nn(data, queries)
        t_bf = time.perf_counter() - t0
        print(f"  brute (x{n_queries}):       {t_bf:8.3f} s  ({t_bf / n_queries * 1e3:.3f} ms/query)")

        vp_ok = np.allclose(vp_dists, bf_dists, atol=1e-6)
        kd_ok = np.allclose(kd_dists, bf_dists, atol=1e-6)
        print(f"  vp correctness:   {'OK' if vp_ok else 'MISMATCH'}  "
              f"(max abs diff = {np.abs(vp_dists - bf_dists).max():.3e})")
        print(f"  kd correctness:   {'OK' if kd_ok else 'MISMATCH'}  "
              f"(max abs diff = {np.abs(kd_dists - bf_dists).max():.3e})")
        if run_lib:
            lib_ok = np.allclose(lib_dists, bf_dists, atol=1e-6)
            print(f"  lib correctness:  {'OK' if lib_ok else 'MISMATCH'}  "
                  f"(max abs diff = {np.abs(lib_dists - bf_dists).max():.3e})")

        rows.append({
            "N": n,
            "vp_build_s": t_vp_build,
            "kd_build_s": t_kd_build,
            "lib_build_s": t_lib_build,
            "vp_query_ms": t_vp / n_queries * 1e3,
            "kd_query_ms": t_kd / n_queries * 1e3,
            "lib_query_ms": t_lib / n_queries * 1e3 if run_lib else float("nan"),
            "brute_query_ms": t_bf / n_queries * 1e3,
            "vp_speedup": t_bf / t_vp,
            "kd_speedup": t_bf / t_kd,
            "lib_speedup": t_bf / t_lib if run_lib else float("nan"),
            "vp_correct": bool(vp_ok),
            "kd_correct": bool(kd_ok),
            "lib_correct": bool(lib_ok) if run_lib else None,
        })

    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(df["N"], df["vp_build_s"], marker="o", label="VP-tree (ours)")
    ax.plot(df["N"], df["kd_build_s"], marker="s", label="KD-tree (scipy)")
    ax.plot(df["N"], df["lib_build_s"], marker="D", label="VP-tree (py lib)")
    ax.set_xlabel("dataset size N")
    ax.set_ylabel("build time (s)")
    ax.set_title("Build time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)

    ax = axes[1]
    ax.plot(df["N"], df["vp_query_ms"], marker="o", label="VP-tree (ours)")
    ax.plot(df["N"], df["kd_query_ms"], marker="s", label="KD-tree (scipy)")
    ax.plot(df["N"], df["lib_query_ms"], marker="D", label="VP-tree (py lib)")
    ax.plot(df["N"], df["brute_query_ms"], marker="^", label="brute force (numpy)")
    ax.set_xlabel("dataset size N")
    ax.set_ylabel("avg query time (ms)")
    ax.set_title("Query time per nearest-neighbor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)

    ax = axes[2]
    ax.plot(df["N"], df["vp_speedup"], marker="o", label="VP-tree (ours)")
    ax.plot(df["N"], df["kd_speedup"], marker="s", label="KD-tree (scipy)")
    ax.plot(df["N"], df["lib_speedup"], marker="D", label="VP-tree (py lib)")
    ax.axhline(1.0, ls="--", color="gray", alpha=0.6)
    ax.set_xlabel("dataset size N")
    ax.set_ylabel("brute / tree query time")
    ax.set_title("Speedup vs. brute force")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.suptitle(f"Trees and bruteforce on MNIST-784  ({int(df['N'].max())} points max), with euclidean distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[500, 1000, 2000, 5000, 10000, 20000])
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "vptree_mnist_perf.png")
    ap.add_argument("--max-lib-n", type=int, default=5000,
                    help="skip pure-Python vptree lib for N above this")
    args = ap.parse_args()

    df = run(args.csv, args.sizes, args.queries, args.seed, args.max_lib_n)
    print("\nSummary:")
    print(df.to_string(index=False))
    df.to_csv(REPO / "vptree_mnist_perf.csv", index=False)
    plot(df, args.out)


if __name__ == "__main__":
    main()
