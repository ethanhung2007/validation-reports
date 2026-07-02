#!/usr/bin/env python3
"""
gen_test_vectors.py
Generates weight matrices and activation vectors for RTL testbench
verification, plus a golden-model output computed in numpy (int32
accumulation, matching the D-IMC's 26b/32b accumulator semantics).

Output: testbench/vectors/*.hex or *.mem (Verilog $readmemh format)
"""
import numpy as np

def gen_vectors(rows=512, cols=32, seed=0, out_dir="../testbench/vectors"):
    rng = np.random.default_rng(seed)
    weights = rng.integers(-128, 128, size=(rows, cols), dtype=np.int16)
    activations = rng.integers(-128, 128, size=(rows,), dtype=np.int16)

    # Golden MVM result (INT8 x INT8 -> INT32 accumulate, matches D-IMC output width)
    golden = (weights.astype(np.int32).T @ activations.astype(np.int32))

    print(f"weights shape: {weights.shape}")
    print(f"activations shape: {activations.shape}")
    print(f"golden output shape: {golden.shape}")
    print(f"golden sample: {golden[:5]}")

    # TODO: write out $readmemh-compatible hex files for RTL testbench

if __name__ == "__main__":
    gen_vectors()
