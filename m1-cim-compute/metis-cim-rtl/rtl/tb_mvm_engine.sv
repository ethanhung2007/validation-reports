// ============================================================
// tb_mvm_engine.sv
// Integration test for mvm_engine: loads randomized weights across
// all banks, runs one full vector-matrix multiply, and checks every
// global output column against a golden model computed by reading
// back the actual DUT storage (same strategy as tb_imc_bank.sv --
// see that file's header for why: it avoids an entire class of
// testbench-side shadow-array bugs).
// ============================================================

module tb_mvm_engine;

    localparam int ROWS          = 16;
    localparam int COLS_PER_BANK = 4;
    localparam int NUM_BANKS     = 4;
    localparam int TOTAL_COLS    = NUM_BANKS * COLS_PER_BANK;
    localparam int WEIGHT_SETS   = 4;
    localparam int WBITS         = 8;
    localparam int ABITS         = 8;
    localparam int ACC_WIDTH     = 26;

    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;

    logic                              wr_en;
    logic [$clog2(NUM_BANKS)-1:0]      wr_bank;
    logic [$clog2(ROWS)-1:0]           wr_row;
    logic [$clog2(COLS_PER_BANK)-1:0]  wr_col;
    logic [$clog2(WEIGHT_SETS)-1:0]    wr_set;
    logic signed [WBITS-1:0]           wr_data;

    logic                              start;
    logic [$clog2(WEIGHT_SETS)-1:0]    active_set;
    logic [ROWS*ABITS-1:0]             act_vec_flat;

    logic [TOTAL_COLS*ACC_WIDTH-1:0]   result_flat;
    logic                              done;
    logic                              busy;

    mvm_engine #(
        .ROWS(ROWS), .COLS_PER_BANK(COLS_PER_BANK), .NUM_BANKS(NUM_BANKS),
        .WEIGHT_SETS(WEIGHT_SETS), .WBITS(WBITS), .ABITS(ABITS)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en), .wr_bank(wr_bank), .wr_row(wr_row), .wr_col(wr_col),
        .wr_set(wr_set), .wr_data(wr_data),
        .start(start), .active_set(active_set), .act_vec_flat(act_vec_flat),
        .result_flat(result_flat), .done(done), .busy(busy)
    );

    int errors = 0;

    // Golden weight readback: global column c lives in bank c/COLS_PER_BANK,
    // local column c%COLS_PER_BANK, inside that bank's weight_cell_array.
    // Division/modulo by a *parameter* (compile-time constant) is fine --
    // only division/modulo by a runtime *signal* would be suspect here,
    // and COLS_PER_BANK is a localparam, not a signal.
    function automatic logic signed [WBITS-1:0] extract_weight(int r, int c_global);
        int bank_idx, local_col, idx;
        logic [WBITS-1:0] raw;
        bank_idx  = c_global / COLS_PER_BANK;
        local_col = c_global % COLS_PER_BANK;
        idx       = r * COLS_PER_BANK + local_col;
        case (bank_idx)
            0: raw = (dut.g_bank[0].u_bank.weights_flat >> (idx*WBITS)) & {WBITS{1'b1}};
            1: raw = (dut.g_bank[1].u_bank.weights_flat >> (idx*WBITS)) & {WBITS{1'b1}};
            2: raw = (dut.g_bank[2].u_bank.weights_flat >> (idx*WBITS)) & {WBITS{1'b1}};
            3: raw = (dut.g_bank[3].u_bank.weights_flat >> (idx*WBITS)) & {WBITS{1'b1}};
            default: raw = '0;
        endcase
        return raw;
    endfunction

    function automatic int extract_act(int r);
        logic signed [ABITS-1:0] raw;
        raw = (act_vec_flat >> (r*ABITS)) & {ABITS{1'b1}};
        return raw;
    endfunction

    integer golden_result [TOTAL_COLS];

    task automatic compute_golden();
        for (int c = 0; c < TOTAL_COLS; c++) begin
            golden_result[c] = 0;
            for (int r = 0; r < ROWS; r++) begin
                automatic int term = extract_act(r) * extract_weight(r, c);
                golden_result[c] += term;
            end
        end
    endtask

    task automatic load_all_weights();
        wr_set = 0;
        for (int b = 0; b < NUM_BANKS; b++) begin
            for (int r = 0; r < ROWS; r++) begin
                for (int c = 0; c < COLS_PER_BANK; c++) begin
                    automatic int val = $urandom_range(0, 255) - 128;
                    wr_en   = 1;
                    wr_bank = b;
                    wr_row  = r;
                    wr_col  = c;
                    wr_data = val;
                    @(posedge clk);
                    #1;
                end
            end
        end
        wr_en = 0;
        @(posedge clk);
        #1;
    endtask

    task automatic run_mvm();
        int val;
        logic [ROWS*ABITS-1:0] flat_tmp;
        logic signed [ACC_WIDTH-1:0] dut_val;
        logic [ACC_WIDTH-1:0] mask;

        flat_tmp = '0;
        for (int r = 0; r < ROWS; r++) begin
            val = $urandom_range(0, 255) - 128;
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
        for (int c = 0; c < TOTAL_COLS; c++) begin
            dut_val = (result_flat >> (c*ACC_WIDTH)) & mask;
            if (dut_val !== golden_result[c]) begin
                $display("FAIL col=%0d (bank=%0d local=%0d)  dut=%0d  golden=%0d",
                    c, c/COLS_PER_BANK, c%COLS_PER_BANK, dut_val, golden_result[c]);
                errors++;
            end else begin
                $display("PASS col=%0d (bank=%0d local=%0d)  dut=%0d  golden=%0d",
                    c, c/COLS_PER_BANK, c%COLS_PER_BANK, dut_val, golden_result[c]);
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
        #1;

        $display("=== Full crossbar test: %0d banks x %0d cols = %0d total outputs ===",
                  NUM_BANKS, COLS_PER_BANK, TOTAL_COLS);
        load_all_weights();
        run_mvm();

        $display("=== Second MVM, same weights, new activations ===");
        run_mvm();

        if (errors == 0)
            $display("\n*** ALL %0d CHECKS PASSED ***", TOTAL_COLS*2);
        else
            $display("\n*** %0d CHECKS FAILED ***", errors);

        $finish;
    end

    initial begin
        #20000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
