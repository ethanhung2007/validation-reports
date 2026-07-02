.PHONY: test bench clean-reports

test:
	cd testbench && iverilog -g2012 -o ../sim/tb_imc_bank.vvp ../rtl/weight_cell_array.sv ../rtl/bitserial_mac.sv ../rtl/bank_accumulator.sv ../rtl/imc_bank.sv tb_imc_bank.sv
	cd testbench && vvp ../sim/tb_imc_bank.vvp
	cd testbench && iverilog -g2012 -o ../sim/tb_mvm_engine.vvp ../rtl/weight_cell_array.sv ../rtl/bitserial_mac.sv ../rtl/bank_accumulator.sv ../rtl/imc_bank.sv ../rtl/mvm_engine.sv tb_mvm_engine.sv
	cd testbench && vvp ../sim/tb_mvm_engine.vvp
	cd testbench && iverilog -g2012 -o ../sim/sweep_harness.vvp ../rtl/weight_cell_array.sv ../rtl/bitserial_mac.sv ../rtl/bank_accumulator.sv ../rtl/imc_bank.sv ../rtl/mvm_engine.sv sweep_harness.sv
	cd testbench && vvp ../sim/sweep_harness.vvp

bench:
	python3 scripts/run_validation.py

clean-reports:
	rm -rf reports
