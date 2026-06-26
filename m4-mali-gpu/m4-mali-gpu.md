# M4 Mali G610 GPU

The validation tests for the Mali G610 GPU was done using Mali Offline Compiler to run benchmarks. Due to the clock frequency of the physical G610 not being specified and using different OpenCL kernels, the validation tests for the Mali G610 GPU is meant to compare and capture the overall linear complexity rather than accurate measurements.

## Checks

Since not many operations are meant to be run on the GPU, the only tests done were attention decode operations. Specifically, single-head QK^T + S·V at kv ∈ {128, 512, 1024} in FP16 — the one operation where GPU offload is structurally justified over CIM.
