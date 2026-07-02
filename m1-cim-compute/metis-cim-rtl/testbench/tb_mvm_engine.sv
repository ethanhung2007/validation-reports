// Canonical testbench lives under rtl/ for now because it uses
// hierarchical DUT readback during bring-up. Keep this wrapper so the
// documented testbench/ build commands compile the real test.
`include "../rtl/tb_mvm_engine.sv"
