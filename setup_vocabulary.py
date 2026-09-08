"""
setup_vocabulary.py — one-time fetch + convert of the real ORB vocabulary.

Run this once:  python3 setup_vocabulary.py

Downloads the official ORB-SLAM3 pre-trained vocabulary (ORBvoc.txt.tar.gz,
~42MB, from github.com/UZ-SLAMLab/ORB_SLAM3 -- the same vocabulary real
ORB-SLAM3 itself ships with), parses its DBoW2 text format, and caches it
as a compressed .npz (~31MB) for fast loading (~0.4s vs ~11s to re-parse
the raw text every startup).

NOT bundled as a pre-built file in this project's delivery zip -- 30MB+
binary caches don't belong in a code drop, and re-running this script is
a one-time, ~90-second cost on a real machine with internet access.

FORMAT (reverse-engineered and verified against the actual file, not
assumed from documentation -- see PROGRESS.md's Phase 5 section for the
verification methodology):
  Header line: "k L scoring_type weighting_type" (k=10, L=6 for the
  standard ORB-SLAM3 vocabulary: a 10-ary tree, 6 levels deep).
  Each following line: "parent_id word_id desc[0..31] weight"
    parent_id: 0 = virtual root; otherwise 1-indexed node id (this row's
               own node id is its line number, 1-indexed).
    word_id:   0 for inner nodes; nonzero (sequential) for leaf/word nodes.
    desc:      32 bytes (0-255), the node's representative ORB descriptor.
    weight:    IDF weight, meaningful only for leaf/word nodes (0 for
               inner nodes, and also legitimately 0 for very common words).
  Verified: children of a given parent are ALWAYS stored as a contiguous
  run of rows (checked exhaustively across all 1,082,072 rows, zero
  violations) -- this is what allows an O(1)-per-level array-based tree
  instead of a slow reverse-lookup structure.
"""

import os
import sys
import time
import tarfile
import urllib.request

import numpy as np

VOCAB_URL = "https://raw.githubusercontent.com/UZ-SLAMLab/ORB_SLAM3/master/Vocabulary/ORBvoc.txt.tar.gz"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "orbvoc_cache.npz")


def parse_orbvoc_txt(path):
    parent, word_id, descs, weight = [], [], [], []
    with open(path) as f:
        header = f.readline().split()
        k, L = int(header[0]), int(header[1])
        for line in f:
            parts = line.split()
            parent.append(int(parts[0]))
            word_id.append(int(parts[1]))
            descs.append([int(x) for x in parts[2:34]])
            weight.append(float(parts[34]))
    return (k, L, np.array(parent, dtype=np.int32), np.array(word_id, dtype=np.int32),
           np.array(descs, dtype=np.uint8), np.array(weight, dtype=np.float32))


def build_child_index(parent):
    """
    child_start[node_id] / child_count[node_id]: O(1) lookup of a node's
    children, exploiting the verified contiguity property (see module
    docstring). node_id=0 is the virtual root; node ids 1..N are rows
    0..N-1 (node_id = row_index + 1).
    """
    n_nodes = len(parent)
    change_points = np.where(np.diff(parent) != 0)[0] + 1
    run_starts = np.concatenate([[0], change_points])
    run_ends = np.concatenate([change_points, [n_nodes]])
    run_parent = parent[run_starts]

    child_start = np.full(n_nodes + 1, -1, dtype=np.int32)
    child_count = np.zeros(n_nodes + 1, dtype=np.uint16)
    for s, e, p in zip(run_starts, run_ends, run_parent):
        child_start[p] = s
        child_count[p] = e - s
    return child_start, child_count


def main():
    if os.path.exists(CACHE_PATH):
        print(f"{CACHE_PATH} already exists. Delete it first to rebuild.")
        return

    txt_path = "/tmp/ORBvoc.txt"
    tar_path = "/tmp/ORBvoc.txt.tar.gz"

    if not os.path.exists(txt_path):
        print(f"Downloading {VOCAB_URL} ...")
        t0 = time.time()
        urllib.request.urlretrieve(VOCAB_URL, tar_path)
        print(f"  done in {time.time()-t0:.1f}s ({os.path.getsize(tar_path)/1e6:.1f} MB)")

        print("Extracting ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall("/tmp")

    print("Parsing (one-time, ~11s for the full vocabulary) ...")
    t0 = time.time()
    k, L, parent, word_id, descs, weight = parse_orbvoc_txt(txt_path)
    print(f"  parsed {len(parent)} nodes in {time.time()-t0:.1f}s (k={k}, L={L})")

    print("Building tree index ...")
    child_start, child_count = build_child_index(parent)

    print(f"Saving cache to {CACHE_PATH} ...")
    np.savez_compressed(CACHE_PATH, k=k, L=L, parent=parent, word_id=word_id,
                        descs=descs, weight=weight,
                        child_start=child_start, child_count=child_count)
    print(f"  done ({os.path.getsize(CACHE_PATH)/1e6:.1f} MB)")
    print("\nSetup complete. orb_vocabulary.py will load this cache automatically.")


if __name__ == "__main__":
    main()
