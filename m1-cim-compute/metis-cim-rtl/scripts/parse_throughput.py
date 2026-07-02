#!/usr/bin/env python3
"""
parse_throughput.py
Parses cycle-count logs from sweep_harness.sv simulation runs,
converts to GOP/s, and fits the same functional form used in the
CIM simulator's silicon-calibrated model:

    G_eff(N, K) = Gmax * N/(N + Na) * K/(K + Kb)

Compares fitted Na/Kb/Gmax against simulator/models/params/m1_cim.json
to check whether the FPGA-measured saturation shape matches silicon.
"""
import numpy as np
# from scipy.optimize import curve_fit  # if available

def geff_model(NK, Gmax, Na, Kb):
    N, K = NK
    return Gmax * (N / (N + Na)) * (K / (K + Kb))

def main():
    # TODO: load cycle-count sweep results (from sim log or CSV)
    # TODO: convert cycles -> GOP/s given clock freq
    # TODO: curve_fit against geff_model
    # TODO: compare against silicon-calibrated Gmax=333.67, Na=577.2, Kb=574.1
    pass

if __name__ == "__main__":
    main()
