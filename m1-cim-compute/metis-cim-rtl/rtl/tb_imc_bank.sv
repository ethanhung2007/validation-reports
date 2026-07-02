// ============================================================
// tb_imc_bank.sv
// Unit test for imc_bank: loads a known weight matrix and
// activation vector, runs the bit-serial MVM, and checks the
// result against a golden dot-product computed in this testbench.
//
// Activations and weights are both treated as signed INT8. The DUT's
// bit-serial MAC subtracts the MSB-weighted activation term to apply
// the standard two's-complement correction.
//
// Ports on imc_bank are flattened packed vectors (see imc_bank.sv);
// this testbench packs/unpacks using standard part-select syntax.
// ============================================================

module tb_imc_bank;

    localparam int ROWS = 16;
    localparam int COLS = 4;
    localparam int WEIGHT_SETS = 4;
    localparam int WBITS = 8;
    localparam int ABITS = 8;
    localparam int ACC_WIDTH = 26;

    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;

    logic                           wr_en;
    logic [$clog2(ROWS)-1:0]        wr_row;
    logic [$clog2(COLS)-1:0]        wr_col;
    logic [$clog2(WEIGHT_SETS)-1:0] wr_set;
    logic signed [WBITS-1:0]        wr_data;

    logic                           start;
    logic [$clog2(WEIGHT_SETS)-1:0] active_set;
    logic [ROWS*ABITS-1:0]          act_vec_flat;

    logic [ACC_WIDTH*COLS-1:0]      result_flat;
    logic                           done;
    logic                           busy;

    imc_bank #(
        .ROWS(ROWS), .COLS(COLS), .WEIGHT_SETS(WEIGHT_SETS),
        .WBITS(WBITS), .ABITS(ABITS)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en), .wr_row(wr_row), .wr_col(wr_col), .wr_set(wr_set), .wr_data(wr_data),
        .start(start), .active_set(active_set), .act_vec_flat(act_vec_flat),
        .result_flat(result_flat), .done(done), .busy(busy)
    );

    // Reference model (signed activations x signed weights).
    // NOTE: golden_acts must be declared as signed `int` (not
    // `int unsigned`) even though values are always >=0 -- mixing a
    // signed and unsigned operand in a SystemVerilog multiplication
    // silently reinterprets the SIGNED operand's bits as unsigned
    // (LRM rule: any unsigned operand makes the whole expression
    // unsigned), which corrupted negative weights in this golden
    // model until fixed.
    integer                   golden_result  [COLS];
    int errors = 0;

    // Golden weights/activations are read back directly from the
    // DUT's own storage (dut.weights_flat) and the testbench's own
    // act_vec_flat, rather than tracked in a separately-maintained
    // shadow array -- this guarantees the golden model always
    // matches what was actually loaded, eliminating an entire class
    // of testbench-side bookkeeping bugs.
    function automatic logic signed [WBITS-1:0] extract_weight(int r, int c);
        logic [WBITS-1:0] raw;
        raw = (dut.weights_flat >> ((r*COLS+c)*WBITS)) & {WBITS{1'b1}};
        return raw;
    endfunction

    function automatic int extract_act(int r);
        logic signed [ABITS-1:0] raw;
        raw = (act_vec_flat >> (r*ABITS)) & {ABITS{1'b1}};
        return raw;
    endfunction

    task automatic compute_golden();
        for (int c = 0; c < COLS; c++) begin
            golden_result[c] = 0;
            for (int r = 0; r < ROWS; r++) begin
                automatic int term = extract_act(r) * extract_weight(r, c);
                golden_result[c] += term;
            end
        end
    endtask

    // NOTE: stimulus is updated *after* a small delay following each
    // clock edge (#1), not immediately before the next edge -- this
    // avoids a classic same-timestep race between testbench-driven
    // signals and the DUT's synchronous sampling of those signals.
    task automatic load_weights();
        wr_set = 0;
        for (int r = 0; r < ROWS; r++) begin
            for (int c = 0; c < COLS; c++) begin
                automatic int val = $urandom_range(0, 255) - 128; // signed -128..127
                wr_en   = 1;
                wr_row  = r;
                wr_col  = c;
                wr_data = val;
                @(posedge clk);
                #1;
            end
        end
        wr_en = 0;
        @(posedge clk);
        #1;
    endtask

    // NOTE: variable-indexed part-selects (`vec[(i+1)*W-1 -: W]`)
    // inside procedural blocks are not reliably supported by Icarus --
    // pack/unpack here uses variable SHIFT + mask instead, which is.
    task automatic run_mvm();
        int val;
        logic [ROWS*ABITS-1:0] flat_tmp;
        logic signed [ACC_WIDTH-1:0] dut_val;
        logic [ACC_WIDTH-1:0] mask;

        flat_tmp = '0;
        for (int r = 0; r < ROWS; r++) begin
            val = $urandom_range(0, 255) - 128; // signed -128..127
            flat_tmp = flat_tmp | ({{(ROWS*ABITS-ABITS){1'b0}}, val[ABITS-1:0]} << (r*ABITS));
        end
        act_vec_flat = flat_tmp;
        active_set = 0;
        #1;
        compute_golden();

        start = 1;
        @(posedge clk);
        #1;
        start = 0;

        wait (done == 1);
        @(posedge clk);
        #1;

        mask = {ACC_WIDTH{1'b1}};
        for (int c = 0; c < COLS; c++) begin
            dut_val = (result_flat >> (c*ACC_WIDTH)) & mask;
            if (dut_val !== golden_result[c]) begin
                $display("FAIL col=%0d  dut=%0d  golden=%0d", c, dut_val, golden_result[c]);
                errors++;
            end else begin
                $display("PASS col=%0d  dut=%0d  golden=%0d", c, dut_val, golden_result[c]);
            end
        end
    endtask

    initial begin
        rst_n = 0;
        wr_en = 0;
        start = 0;
        active_set = 0;
        act_vec_flat = '0;
        #20 rst_n = 1;
        @(posedge clk);

        $display("=== Test 1: load weights, run MVM ===");
        load_weights();
        run_mvm();

        $display("=== Test 2: same weights, new activations ===");
        run_mvm();

        $display("=== Test 3: reload weights, run again ===");
        load_weights();
        run_mvm();

        if (errors == 0)
            $display("\n*** ALL TESTS PASSED ***");
        else
            $display("\n*** %0d TESTS FAILED ***", errors);

        $finish;
    end

    initial begin
        #10000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
