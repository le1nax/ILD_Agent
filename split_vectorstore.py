"""
Utility: splits ild_index/default__vector_store.json into 5 parts
so each part stays under GitHub's 100 MB file size limit.

Usage:
    python split_vectorstore.py          # split only
    python split_vectorstore.py --verify # split, then simulate cold-start reassembly
                                         # and verify the result matches the original

The parts are written alongside the original:
    ild_index/default__vector_store.json.part1  ...  .part5

At runtime, graph.py reassembles them automatically before loading the index.
"""

import hashlib
import math
import os
import shutil
import sys

VECTOR_STORE_PATH = os.path.join("ild_index", "default__vector_store.json")
NUM_PARTS = 5


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split():
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"ERROR: File not found: {VECTOR_STORE_PATH}")
        sys.exit(1)

    size = os.path.getsize(VECTOR_STORE_PATH)
    part_size = math.ceil(size / NUM_PARTS)
    print(f"Splitting {VECTOR_STORE_PATH}")
    print(f"  Total size : {size / 1024 / 1024:.1f} MB")
    print(f"  Parts      : {NUM_PARTS} × ~{part_size / 1024 / 1024:.1f} MB each")

    with open(VECTOR_STORE_PATH, "rb") as f:
        for i in range(1, NUM_PARTS + 1):
            chunk = f.read(part_size)
            if not chunk:
                break
            part_path = VECTOR_STORE_PATH + f".part{i}"
            with open(part_path, "wb") as out:
                out.write(chunk)
            print(f"  Wrote {part_path}  ({len(chunk) / 1024 / 1024:.1f} MB)")

    print("Split complete.\n")


def verify():
    """Simulate a cold-start: hide the full JSON, reassemble from parts, compare."""
    if not os.path.exists(VECTOR_STORE_PATH):
        print("ERROR: Original file not found — run split first.")
        sys.exit(1)

    part1 = VECTOR_STORE_PATH + ".part1"
    if not os.path.exists(part1):
        print("ERROR: No .part1 found — run split first.")
        sys.exit(1)

    print("=== Sanity check: simulating cold-start reassembly ===\n")

    # 1. Hash the original
    print("Step 1/4  Computing SHA-256 of original...")
    original_hash = _sha256(VECTOR_STORE_PATH)
    print(f"          {original_hash}\n")

    # 2. Move the original aside
    backup_path = VECTOR_STORE_PATH + ".backup"
    print("Step 2/4  Moving original aside (simulating missing file on fresh clone)...")
    shutil.move(VECTOR_STORE_PATH, backup_path)

    # 3. Reassemble using the same logic as graph.py
    print("Step 3/4  Reassembling from parts...")
    i = 1
    with open(VECTOR_STORE_PATH, "wb") as out:
        while True:
            part_path = VECTOR_STORE_PATH + f".part{i}"
            if not os.path.exists(part_path):
                break
            with open(part_path, "rb") as part:
                out.write(part.read())
            print(f"          Merged {part_path}")
            i += 1
    assembled_size = os.path.getsize(VECTOR_STORE_PATH)
    print(f"          Assembled {i - 1} parts → {assembled_size / 1024 / 1024:.1f} MB\n")

    # 4. Compare hashes
    print("Step 4/4  Verifying SHA-256...")
    assembled_hash = _sha256(VECTOR_STORE_PATH)
    print(f"          Original   : {original_hash}")
    print(f"          Reassembled: {assembled_hash}\n")

    if original_hash == assembled_hash:
        print("✓  PASS — reassembled file is byte-for-byte identical to the original.")
        os.remove(backup_path)
    else:
        print("✗  FAIL — hashes differ! Restoring original from backup.")
        os.remove(VECTOR_STORE_PATH)
        shutil.move(backup_path, VECTOR_STORE_PATH)
        sys.exit(1)


if __name__ == "__main__":
    verify_mode = "--verify" in sys.argv
    split()
    if verify_mode:
        verify()
