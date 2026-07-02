// ============================================================
// sweep_harness.sv
// Phase C: throughput characterization, INCLUDING weight-load time.
//
// Earlier version of this harness only timed the bit-serial compute
// phase (weights pre-loaded, assumed free) and found cycle count is
// constant regardless of active K -- a genuine finding, but it meant
// this harness could not reproduce ANY K/N-dependent throughput curve,
// because the one real K/N-dependent cost in this design (weight
// loading) was excluded from the measurement.
//
// This version times BOTH phases for each (K,N) point:
//   1. Weight load: imc_bank's weight_cell_array has a SINGLE
//      (row,col) write port (see weight_cell_array.sv) -- one cell
//      per cycle, a real structural constraint of this RTL, not an
//      assumption invented for this harness. Loading only the ACTIVE
//      K x N submatrix (not the full fixed physical array) costs
//      K*N cycles. This matches the docs/README.md explanation of
//      why small N/K "wastes weight-load bandwidth proportionally"
//      on real silicon.
//   2. Compute: fixed BITSERIAL_DEPTH+1 cycles regardless of K/N (the
//      Phase C finding from the previous version, still true and
//      still reported below).
//
// total_cycles(K,N) = K*N (load) + BITSERIAL_DEPTH+1 (compute)
//
// This DOES produce a genuine, RTL-derived, monotonic-saturating
// throughput curve in both K and N (fixed cost amortized over more
// work -> diminishing returns), which is the qualitative mechanism
// behind the silicon G_eff(N,K) curve. It will NOT numerically match
// the silicon-fit Gmax=333.67/Na=577.2/Kb=574.1 -- those come from a
// 512x512 quad-core chip at 800MHz with its own (unpublished) DMA
// architecture, vs. this toy 64x32 single-port sim model. The claim
// here is "same amortization mechanism, RTL-verified," not "same
// numbers." scripts/run_validation.py fits this RTL data independently
// and validates shape properties on it (not on a copy of the silicon
// parameters).
//
// A second sweep (weights loaded ONCE, M compute passes reused) times
// the prefill-style "load once, decode M times" pattern, giving a
// genuinely RTL-derived affine latency-vs-M curve.
// ============================================================

module sweep_harness;

    // Fixed physical crossbar size for this harness run (compile-time).
    localparam int ROWS          = 64;
    localparam int COLS_PER_BANK = 8;
    localparam int NUM_BANKS     = 4;
    localparam int TOTAL_COLS    = NUM_BANKS * COLS_PER_BANK;
    localparam int WEIGHT_SETS   = 4;
    localparam int WBITS         = 8;
    localparam int ABITS         = 8;
    localparam int BITSERIAL_DEPTH = ABITS;
    localparam int ACC_WIDTH     = 26;

    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;   // period irrelevant for cycle counting; only #cycles matters

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

    // Free-running cycle counter -- tasks below record its value before
    // and after an operation and report the delta. Simpler and more
    // robust than a dedicated start/done counter when the same counter
    // must span both the weight-load loop and the compute handshake.
    integer sim_cycle;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) sim_cycle <= 0;
        else        sim_cycle <= sim_cycle + 1;
    end

    // Load the active K_active x N_active submatrix (global column
    // indices 0..N_active-1, mapped to bank/local-col exactly as
    // tb_mvm_engine.sv's extract_weight does). One write per cycle --
    // the weight_cell_array write port only accepts one (row,col) per
    // cycle, so this loop's cycle count IS the load cost, not a
    // simulated approximation of it.
    task automatic load_active(input int K_active, input int N_active, output int cycles_out);
        int start_cyc;
        int bank_idx, local_col;
        start_cyc = sim_cycle;
        for (int r = 0; r < K_active; r++) begin
            for (int gc = 0; gc < N_active; gc++) begin
                bank_idx  = gc / COLS_PER_BANK;
                local_col = gc % COLS_PER_BANK;
                wr_en   = 1;
                wr_bank = bank_idx[$clog2(NUM_BANKS)-1:0];
                wr_row  = r[$clog2(ROWS)-1:0];
                wr_col  = local_col[$clog2(COLS_PER_BANK)-1:0];
                wr_set  = 0;
                wr_data = 8'sd1;
                @(posedge clk); #1;
            end
        end
        wr_en = 0;
        cycles_out = sim_cycle - start_cyc;
    endtask

    // Run one MVM with K_active rows driven nonzero (rows >= K_active
    // are zero -- contribute nothing, modeling a smaller K within the
    // fixed physical array) and measure cycles to `done`.
    task automatic run_compute(input int K_active, output int cycles_out);
        int start_cyc;
        logic [ROWS*ABITS-1:0] flat_tmp;
        flat_tmp = '0;
        for (int r = 0; r < K_active; r++) begin
            flat_tmp = flat_tmp | (32'(1) << (r*ABITS));  // simple nonzero pattern
        end
        act_vec_flat = flat_tmp;
        active_set = 0;

        start_cyc = sim_cycle;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        wait (done == 1);
        @(posedge clk); #1;

        cycles_out = sim_cycle - start_cyc;
    endtask

    int load_cyc, compute_cyc, k_val, n_val;
    int k_grid[6];
    int n_grid[6];

    initial begin
        rst_n = 0; wr_en = 0; start = 0;
        #20 rst_n = 1;
        @(posedge clk); #1;

        $display("========================================================");
        $display(" G_eff(N,K) sweep -- RTL-derived (load + compute cycles)");
        $display(" Fixed physical array: ROWS=%0d COLS=%0d (%0d banks x %0d)",
                   ROWS, TOTAL_COLS, NUM_BANKS, COLS_PER_BANK);
        $display(" Load: 1 cycle/weight cell (single write port, real HW");
        $display(" constraint). Compute: BITSERIAL_DEPTH=%0d, fixed.", BITSERIAL_DEPTH);
        $display("========================================================");

        k_grid = '{4, 8, 16, 32, 48, 64};
        n_grid = '{2, 4, 8, 16, 24, 32};

        for (int ki = 0; ki < 6; ki++) begin
            for (int ni = 0; ni < 6; ni++) begin
                k_val = k_grid[ki];
                n_val = n_grid[ni];
                load_active(k_val, n_val, load_cyc);
                run_compute(k_val, compute_cyc);
                $display("SWEEPPT K=%0d N=%0d LOAD_CYCLES=%0d COMPUTE_CYCLES=%0d TOTAL_CYCLES=%0d",
                          k_val, n_val, load_cyc, compute_cyc, load_cyc + compute_cyc);
            end
        end

        $display("");
        $display("FINDING: compute-only latency is CONSTANT (%0d cycles)", compute_cyc);
        $display("regardless of active K/N -- every row's bit-serial");
        $display("contribution happens in the SAME cycle (parallel across");
        $display("rows, serial only over BITSERIAL_DEPTH activation bits).");
        $display("The K/N-dependence above comes entirely from weight-LOAD");
        $display("cycles (K*N, one cell/cycle through the single write port),");
        $display("not from compute. Same qualitative mechanism as the silicon");
        $display("G_eff(N,K) curve (fixed cost amortized over more work) --");
        $display("see scripts/run_validation.py for the independent RTL-data");
        $display("fit and shape checks. Exact Gmax/Na/Kb will NOT match");
        $display("m1_cim.json (different array size/technology/DMA design),");
        $display("only the qualitative saturation shape is being compared.");

        $display("");
        $display("========================================================");
        $display(" Prefill M-sweep -- load once (K=%0d,N=%0d), reuse over M", ROWS, TOTAL_COLS);
        $display("========================================================");

        load_active(ROWS, TOTAL_COLS, load_cyc);
        $display("PREFILL_LOAD_CYCLES=%0d", load_cyc);

        begin
            int m_grid[9];
            int m_val, total_compute_cyc, one_cyc;
            m_grid = '{1, 2, 4, 8, 16, 32, 64, 128, 256};
            for (int mi = 0; mi < 9; mi++) begin
                m_val = m_grid[mi];
                total_compute_cyc = 0;
                for (int m = 0; m < m_val; m++) begin
                    run_compute(ROWS, one_cyc);
                    total_compute_cyc += one_cyc;
                end
                $display("PREFILLPT M=%0d COMPUTE_CYCLES=%0d", m_val, total_compute_cyc);
            end
        end

        $display("");
        $display("SWEEP_HARNESS_DONE OK");
        $finish;
    end

    initial begin
        // Load-cycle sums alone (K*N across the sweep grid, plus the
        // full-tile prefill load and up to M=256 compute passes) run
        // to ~22k clock cycles => ~250k time units at #5 half-period
        // plus #1 stimulus delays; give generous headroom.
        #4000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
