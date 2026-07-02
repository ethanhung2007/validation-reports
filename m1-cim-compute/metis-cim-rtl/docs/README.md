# Metis-Inspired CIM Crossbar RTL — RTL + benchmark-shape validation

RTL model of a D-IMC (digital in-memory computing) MVM engine,
structurally inspired by the Axelera Metis AIPU (ISSCC 2024).

## Status

- **Phase A (done, verified)**: single IMC bank (`imc_bank.sv`) --
  12/12 randomized checks pass against a golden model.
- **Phase B (done, verified)**: full crossbar (`mvm_engine.sv`) --
  4 banks x 4 columns tiled = 16 outputs, 32/32 randomized checks
  pass across two independent MVM runs.
- **Phase C (done, RTL-derived)**: throughput sweep harness
  (`sweep_harness.sv`) times weight-load (real single-write-port
  constraint of `weight_cell_array.sv`, one cycle/cell) plus compute
  across a (K,N) grid and an M-reuse sweep. This produces a genuine,
  RTL-simulated monotonic-saturating G_eff(N,K)-shaped curve and an
  affine load-once/reuse-M-times latency curve -- see "Phase C
  finding" below for what does and does not match the silicon
  numbers.
- **Benchmark validation (done)**: `scripts/run_validation.py` runs
  the RTL tests, fits/validates the G_eff-shape and prefill-affine
  curves from the sweep harness's own cycle-accurate output (not from
  the silicon-calibrated constants), and separately reports the
  residency-cliff shape as an analytic-only assumption (this compute
  RTL has no memory-hierarchy model, so that one curve is NOT
  RTL-validated -- see the report's own callout).

```bash
# One-command RTL + benchmark-shape validation
make bench
# Outputs:
#   reports/benchmark_report.html
#   reports/data/summary.json
#   reports/data/*.csv

# RTL tests only
make test

# Phase A
cd testbench && iverilog -g2012 -o ../sim/tb_imc_bank.vvp \
    ../rtl/weight_cell_array.sv ../rtl/bitserial_mac.sv \
    ../rtl/bank_accumulator.sv ../rtl/imc_bank.sv tb_imc_bank.sv
vvp ../sim/tb_imc_bank.vvp
# Expect: *** ALL TESTS PASSED ***

# Phase B
iverilog -g2012 -o ../sim/tb_mvm_engine.vvp \
    ../rtl/weight_cell_array.sv ../rtl/bitserial_mac.sv \
    ../rtl/bank_accumulator.sv ../rtl/imc_bank.sv ../rtl/mvm_engine.sv \
    tb_mvm_engine.sv
vvp ../sim/tb_mvm_engine.vvp
# Expect: *** ALL 32 CHECKS PASSED ***

# Phase C
iverilog -g2012 -o ../sim/sweep_harness.vvp \
    ../rtl/weight_cell_array.sv ../rtl/bitserial_mac.sv \
    ../rtl/bank_accumulator.sv ../rtl/imc_bank.sv ../rtl/mvm_engine.sv \
    sweep_harness.sv
vvp ../sim/sweep_harness.vvp
```

## Phase C finding (important -- read before using this for validation)

The sweep harness times two separate things per (K,N) point:

1. **Compute** (bit-serial MVM, weights already loaded): **constant
   (10 cycles) regardless of active K or N.** In this bit-serial
   design, all ROWS contribute to each column's dot product in the
   SAME cycle (the crossbar's physical parallelism); the only
   serialization is over the BITSERIAL_DEPTH=8 activation bits. This
   part of the original Phase C finding still holds and is still a
   real, correct fact about the architecture as built, not a bug.
2. **Weight load**: `weight_cell_array.sv` has a single (row, col)
   write port (see its header) -- a real structural constraint, not
   an assumption invented for this harness. Loading an active K x N
   submatrix costs exactly K*N cycles. This is the part the original
   harness excluded ("weights assumed pre-loaded and free"), and it's
   the only K/N-dependent cost in this design.

Adding load time back in produces `total_cycles(K,N) = K*N + 10`, and
`scripts/run_validation.py` derives `ops_per_cycle = 2*K*N/total_cycles`
from that -- a genuine, RTL-simulated curve that **rises monotonically
with N, saturates with N, and rises monotonically with K**, checked
directly against this sweep data (not asserted by construction). That
is the same *qualitative mechanism* (a fixed per-operation cost
amortized over more work) behind the silicon-calibrated G_eff(N,K)
curve (`Gmax=333.67, Na=577.2, Kb=574.1` from `m1_cim.json`).

**What still does NOT match**: the RTL's load cost scales with the
*product* K*N (a single serial write port touches both dimensions
equally), so `ops_per_cycle` is a function of K*N alone -- a genuinely
different rational function from the silicon's *separable*
`N/(N+Na) * K/(K+Kb)` form (which implies independent per-axis
half-saturation constants). Fitting the RTL data to that separable
form gets R²≈0.76 (reported, not gated on a pass/fail threshold,
because the mismatch is structural, not noise): both curves
diminishing-returns-saturate, but they are not the identical function.
Reproducing the silicon's exact separable shape would need a weight
load path that is parallel across one axis and serial across the
other (e.g. a per-column bus), which isn't how `weight_cell_array.sv`
is built. Exact Gmax/Na/Kb magnitude was never expected to match
regardless -- different array size (64x32 here vs 512x512 quad-core),
technology, and clock.

The **prefill M-sweep** (load a full tile once, run M compute passes)
fits the RTL cycle data to an affine model exactly: `total_cycles(M) =
2048 + 10*M`, R²=1.0 -- 2048 = K*N load cycles for the full tile, 10 =
the fixed compute-cycle cost. This is a real affine relationship
measured from simulation, matching the qualitative "load once, reuse
over M" latency shape the docs describe for prefill.

The **residency cliff** (SRAM-resident vs DRAM-spill) is NOT covered
by any of this: it's a memory-hierarchy effect, and this repo's
compute RTL has no SRAM/DRAM model to derive or check it against.
`scripts/run_validation.py` still reports that curve using the
silicon-calibrated constants, but labels it explicitly as an
analytic-only assumption and excludes it from the RTL-validated
PASS/FAIL gate in `reports/benchmark_report.html`.

## Architecture (matches Metis ISSCC 2024, Fig 11.3.2/11.3.4)

- **ROWS x COLS crossbar**: rows = inputs, columns = independent
  output dot-products. Every row-column cell holds an 8-bit signed
  weight, in one of WEIGHT_SETS (default 4) double-buffered slots.
- **Bit-serial activation feed**: signed INT8 activations arrive 1
  bit/cycle (LSB first) across BITSERIAL_DEPTH (=8) cycles. Each
  column accumulates `sum_r(act_bit[r] * weight[r][col]) << bit_idx`
  for lower bits and subtracts the MSB-weighted term for the
  two's-complement sign bit. After 8 cycles the column holds the full
  signed INT8xINT8 dot product in a 26b accumulator.

## Hard-won Icarus Verilog lessons (documented so you don't re-hit these)

These cost real debugging time and are worth knowing before writing
more RTL for this project:

1. **Unpacked array module ports silently return X.** `output logic
   [W-1:0] foo [N]` (array-of-vectors) crossing a module port does
   not reliably propagate in Icarus 12.0, regardless of whether it's
   driven by `always_comb`, `always @*`, or `generate`+`assign`.
   Fix: flatten to a single packed vector (`output logic
   [N*W-1:0] foo_flat`) and unpack internally with bit-slicing.
   Unpacked arrays are fine as *internal* signals -- only the port
   crossing is broken.

2. **Variable-indexed part-selects inside procedural blocks are
   unreliable.** `vec[(i+1)*W-1 -: W] = x;` with a runtime variable
   `i` inside an `always`/task can silently write "all bits" instead
   of the intended slice. Constant/genvar-indexed part-selects are
   fine. Fix: use variable *shift* + mask instead
   (`vec = vec | (x << (i*W))`), which Icarus handles correctly, or
   do the indexing via a `generate`/genvar loop with constant
   per-instance offsets.

3. **Variable bit-select on the LHS of a packed vector** (`vec[i] =
   bit;` with runtime `i`) has the same issue as #2. Fix: compute
   into an internal *unpacked* array (variable indexing into unpacked
   arrays works fine) and pack into the vector via a genvar-generate.

4. **Classic same-timestep testbench race.** Changing stimulus
   signals immediately after `@(posedge clk)` and before the next
   edge can race with the DUT's synchronous sampling of those same
   signals, causing every-other write to silently drop. Fix: add a
   small delay (`#1`) after each `@(posedge clk)` before changing
   stimulus.

5. **Mixing signed and unsigned operands in an expression makes the
   whole expression unsigned** (standard Verilog/SV rule, easy to
   forget). If a signed reference value gets multiplied against an
   `int unsigned`, the signed operand's bit pattern is silently
   reinterpreted as unsigned. Declare golden/reference values as
   `int` (signed) even when they happen to always be non-negative.

6. **`golden_result[c] += a * b` without an intermediate typed
   variable computed a different (wrong) value than manually summing
   the same terms with an `automatic int term = a*b;` in between.**
   Root cause not fully isolated (likely an operator-width/promotion
   interaction specific to this Icarus version) -- the practical fix
   is to always materialize multiplication results into an explicitly
   sized intermediate before accumulating, rather than chaining
   `+=` directly against a multiply expression.

## Build phases

- **Phase A (done)** -- single IMC bank, unit-verified against a
  golden model, matches DUT exactly across 12/12 randomized checks.
- **Phase B (done)** -- full crossbar: NUM_BANKS x imc_bank tiled
  (shared activation rows, independent weight/output per bank),
  32/32 randomized checks pass across two MVM runs.
- **Phase C (done, RTL-derived)** -- throughput sweep harness times
  weight-load (real single-write-port constraint) plus compute across
  a (K,N) grid and an M-reuse sweep, producing a genuine
  monotonic-saturating throughput shape and an exact affine
  latency-vs-M fit, both derived from and checked against actual RTL
  simulation cycle counts (see "Phase C finding" above for what does
  and doesn't match the silicon-calibrated numbers).
- **Benchmark validation (done)** -- `make bench` runs RTL correctness
  tests and generates RTL-derived shape-validation artifacts under
  `reports/`; the residency-cliff curve remains an analytic-only
  assumption, clearly labeled and excluded from the PASS/FAIL gate.

## FPGA scoping note

Full 512x512x4-weight-sets will exceed Arty A7-100T BRAM/DSP budget,
and the "full parallel read every cycle" weight storage model
(needed to match the crossbar's physical parallelism) becomes
expensive to synthesize at that scale. All modules are parameterized
(`ROWS`, `COLS`, `WEIGHT_SETS`) -- Phase A/B run at any scale in
simulation; FPGA synthesis should target something like 64x32x1
first and scale up only as resources allow.

## Directory layout

    rtl/         SystemVerilog RTL plus canonical bring-up benches
    testbench/   wrappers for the documented simulation commands
    scripts/     validation runner and helper scripts
    reports/     generated benchmark outputs (ignored by git)
    sim/         simulation logs
    docs/        this file

## Reference

Hager et al., "Metis AIPU: A 12nm 15TOPS/W 209.6TOPS SoC for Cost-
and Energy-Efficient Inference at the Edge," ISSCC 2024, Session 11.3.
