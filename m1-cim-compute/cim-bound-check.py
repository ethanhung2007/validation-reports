"""
AIPU / CIM Tile Roofline Bound Checker
Model parameters from 02-cim.html §2 (m1_cim.json / cim_prefill_fit.json)

Convention: 1 GOP = 2 giga-ops (multiply + add counted separately), consistent
with §2 which defines workload as 2·K·N ops. So 1 MAC = 2 ops = 1 GOP.

Hardware constants:
  - Peak INT8 compute : 214 TOPS (datasheet)
  - On-chip SRAM BW  : ~3.2 TB/s (4 cores × 512×512 crossbar × 800 MHz × 2B)
  - LPDDR4x BW       : 68 GB/s (measured)
  - Knee             : 8.2M params (residency cliff)
  - Spill floor      : 70 GOP/s (memory-bound, DRAM-limited)
"""

import numpy as np

GMAX     = 333.67    # GOP/s fitted saturation
NA       = 577.2
KB       = 574.1
A_US     = 40.8      # prefill intercept µs
B_US     = 0.094     # prefill slope µs/col
KNEE     = 8.2e6     # residency cliff (params = K*N)
SPILL_FLOOR_GOPS = 70.0

PEAK_TOPS   = 214.0
PEAK_GOP_S  = PEAK_TOPS * 1000  # 214,000 GOP/s
SRAM_BW     = 3.2e12            # TB/s → bytes/s
DRAM_BW     = 68e9              # bytes/s
W           = 4 * 512           # = 2048

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠  WARN"

results    = []
violations = []

def geff(n_tile, K):
    return GMAX * (n_tile / (n_tile + NA)) * (K / (K + KB))

def gemv_lat_us(K, N):
    """Tiled GEMV latency: uses G_eff for SRAM tiles, cliff model for spill."""
    lat = 0.0
    remaining = N
    while remaining > 0:
        n_tile = min(remaining, W)
        kn = K * n_tile
        if K * N > KNEE:   # whole GEMV is in spill regime
            ops = 2 * K * n_tile
            lat += (ops / (SPILL_FLOOR_GOPS * 1e9)) * 1e6
        else:
            g = geff(n_tile, K)
            lat += (2 * K * n_tile) / (g * 1e9) * 1e6
        remaining -= n_tile
    return lat

def check(label, lat_us, ops_count, bytes_acc, regime):
    lat_s = lat_us * 1e-6
    bw = DRAM_BW if regime == "spill" else SRAM_BW
    compute_floor_us = (ops_count / (PEAK_GOP_S * 1e9)) * 1e6
    bw_floor_us      = (bytes_acc / bw) * 1e6
    floor_us         = max(compute_floor_us, bw_floor_us)
    bound            = "compute" if compute_floor_us >= bw_floor_us else "BW"
    ok               = lat_us >= floor_us * 0.97   # 3% tolerance
    status           = PASS if ok else FAIL
    if not ok:
        violations.append((label, lat_us, floor_us, bound))
    results.append({"label": label, "lat_us": lat_us, "floor_us": floor_us,
                    "compute_fl": compute_floor_us, "bw_fl": bw_floor_us,
                    "bound": bound, "regime": regime, "status": status})

# ── Check 1: GEMV decode ──────────────────────────────────────────────────
print("=" * 76)
print("CHECK 1 — GEMV decode  (M=1, weight-stationary, cliff-aware)")
print("=" * 76)
print(f"  {'K':>6} {'N':>6} {'K·N':>10} {'regime':<8} {'lat_µs':>10} {'comp_fl':>10} {'bw_fl':>10} status")
print(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} ------")

for K in [2048, 3072, 4096]:
    for N in [64, 256, 512, 1024, 2048, 4096]:
        lat  = gemv_lat_us(K, N)
        ops  = 2 * K * N
        byt  = K * N + K + N   # weights + activations + output (INT8)
        KN   = K * N
        reg  = "spill" if KN > KNEE else "sram"
        check(f"GEMV K={K} N={N}", lat, ops, byt, reg)
        r = results[-1]
        flag = "" if r["status"] == PASS else " ←"
        print(f"  {K:>6} {N:>6} {KN:>10,} {reg:<8} {lat:>10.3f} {r['compute_fl']:>10.4f} {r['bw_fl']:>10.4f} {r['status']}{flag}")
    print()

# ── Check 2: GEMM prefill ─────────────────────────────────────────────────
print("=" * 76)
print("CHECK 2 — GEMM prefill affine  (canonical tile K=N=2048)")
print("=" * 76)
print(f"  Model: tile_lat = {A_US} + {B_US}·M µs  |  full_lat = (K·N/W²)·tile_lat")
print()
print(f"  {'M':>6} {'tile_lat':>12} {'full_lat':>12} {'compute_fl':>12} {'bw_fl':>10} status")
print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*10} ------")

K2, N2 = 2048, 2048
for M in [1, 2, 8, 32, 128, 256, 508]:
    tile_lat = A_US + B_US * M
    full_lat = (K2 * N2 / W**2) * tile_lat
    ops  = 2 * M * K2 * N2
    byt  = (K2 * N2 + M * K2 + M * N2)
    reg  = "spill" if K2 * N2 > KNEE else "sram"
    check(f"GEMM M={M}", full_lat, ops, byt, reg)
    r = results[-1]
    flag = "" if r["status"] == PASS else " ←"
    print(f"  {M:>6} {tile_lat:>12.3f} {full_lat:>12.3f} {r['compute_fl']:>12.5f} {r['bw_fl']:>10.4f} {r['status']}{flag}")

print()

# ── Check 3: Spill floor BW consistency (with correct ops convention) ────
print("=" * 76)
print("CHECK 3 — Spill floor BW consistency")
print("=" * 76)
# 1 GOP = 2 ops (multiply + add), so 1 Giga-MAC = 2 Giga-ops
# Implied DRAM BW for weight-dominant GEMV: BW = MACs/s × bytes/MAC
# = (70e9 ops/s ÷ 2 ops/MAC) × 1 byte/MAC = 35 GB/s
implied_bw = (SPILL_FLOOR_GOPS * 1e9 / 2) * 1  # MACs/s × bytes/MAC
print(f"  Spill floor : {SPILL_FLOOR_GOPS} GOP/s")
print(f"  Convention  : 1 GOP = 2 ops (§2: 2·K·N ops, consistent with §3 measured 89.2 TOPS)")
print(f"  Implied BW  : {SPILL_FLOOR_GOPS} GOP/s ÷ 2 × 1 byte/MAC = {implied_bw/1e9:.1f} GB/s")
print(f"  DRAM BW     : {DRAM_BW/1e9:.0f} GB/s")
if implied_bw <= DRAM_BW:
    print(f"  {PASS} — implied {implied_bw/1e9:.1f} GB/s ≤ {DRAM_BW/1e9:.0f} GB/s physical limit")
    results.append({"label": "Spill floor BW", "status": PASS})
else:
    print(f"  {FAIL} — implied {implied_bw/1e9:.1f} GB/s > {DRAM_BW/1e9:.0f} GB/s")
    violations.append(("Spill floor BW", implied_bw/1e9, DRAM_BW/1e9, "BW"))
    results.append({"label": "Spill floor BW", "status": FAIL})
print()

# ── Check 4: Gmax vs datasheet ───────────────────────────────────────────
print("=" * 76)
print("CHECK 4 — Gmax vs datasheet ceiling")
print("=" * 76)
print(f"  Gmax = {GMAX} GOP/s  vs  214,000 GOP/s (214 TOPS)")
if GMAX <= PEAK_GOP_S:
    util = GMAX / PEAK_GOP_S * 100
    print(f"  {PASS} — Gmax is {util:.3f}% of datasheet ceiling (memory-bound, as expected)")
    results.append({"label": "Gmax vs datasheet", "status": PASS})
else:
    print(f"  {FAIL} — Gmax exceeds datasheet ceiling")
    violations.append(("Gmax vs datasheet", GMAX, PEAK_GOP_S, "compute"))
    results.append({"label": "Gmax vs datasheet", "status": FAIL})
print()

# ── Summary ───────────────────────────────────────────────────────────────
print("=" * 76)
print("SUMMARY")
print("=" * 76)
n_pass = sum(1 for r in results if r["status"] == PASS)
n_fail = sum(1 for r in results if r["status"] == FAIL)
print(f"  Checks : {len(results)}   Pass : {n_pass}   Fail : {n_fail}")
if violations:
    print("\n  Violations:")
    for v in violations:
        print(f"    — {v[0]}")
else:
    print(f"\n  {PASS} All {len(results)} checks passed — no super-physical latencies detected.")
    print()
    print("  Note: Gmax utilization is very low (memory-bound at all measured points),")
    print("  consistent with §2 finding that decode is load-store bottlenecked.")
    print()
    print("  Validation scope: roofline bound checks only. This confirms no physically")
    print("  impossible values exist in the fitted model. It does NOT replace direct")
    print("  hardware measurement (board no longer available).")
