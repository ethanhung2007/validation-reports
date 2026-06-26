# M1 CIM Compute Core

## Disclaimer for the CIM validation

Due to the fact that Axelera doesn't publish microarchicutral details, and that they do not have a simulation software or other testbenches, the validation for the CIM Compute Core simply checks whether the data makes sense or not. For the most part, the testing in this section merely checks whether the fitted data violates any physical laws, indicating that the values are completely false.

The testing script revolves around the physical limits in compute and memory bandwidth. 

```text
Compute is modeled as:
minimum_latency = total_ops / peak_compute

whereas Bandwidth is modeled as 
minimum_latency = total_bytes / peak_bandwidth
```

Note that the minimum latency of any operation needs to be longer than the larger of these two for any operation, else, the data is indicating that the hardware has done something physically impossible.

## Assumptions and Constants

For the testing, the script just uses these assumptions as constants given by the Axelera datasheet or estimated effective values to run in comparison to the recorded data.

```text
These constants include:

PEAK_GOP_S  = 214,000 GOP/s    # 214 TOPS from Axelera datasheet
SRAM_BW     = 3.2 TB/s         # estimated on-chip crossbar bandwidth
DRAM_BW     = 68 GB/s          # measured LPDDR4x ceiling on RK3588
W           = 4 × 512 = 2048   # effective output width (4 cores × 512 crossbar)
KNEE        = 8.2M params       # residency cliff inflection point
SPILL_FLOOR = 70 GOP/s          # throughput after DRAM spill
```

## Checks

### GEMV (decode, M=1)

```text
[µs] G_eff(N, K) = Gmax · N/(N + Na) · K/(K + Kb) [GOP/s]
```

### GEMM (prefill, M>1)

```text
tile_lat(M) = a + b·M # latency of one 2048×2048 tile 
full_GEMM_lat(M, K, N) = (K·N / W²) · tile_lat(M)
```

### Spill Floor Bandwidth

```text
implied_BW = MACs/s × bytes/MAC
           = (70e9 ops/s ÷ 2 ops/MAC) × 1 byte/MAC
           = 35 GB/s
```

### Gmax Ceiling Check

```text
Datasheet provides a maximum of 214 TOPs, whereas the recorded Gmax is 0.334 TOPS
```
