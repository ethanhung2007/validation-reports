// ============================================================
// mvm_engine.sv
// Full crossbar: NUM_BANKS x imc_bank tiled to cover the target
// output width. This is the Phase B deliverable.
//
// Physical picture: all banks share the SAME ROWS-wide activation
// input (the crossbar's rows are common to every bank -- only the
// COLUMNS are partitioned across banks). Each bank independently
// stores its own COLS_PER_BANK-wide slice of the weight matrix and
// computes its own slice of the output vector. All banks run in
// lockstep (same BITSERIAL_DEPTH, same start/done timing), since
// the row-side activation feed is identical for all of them.
//
// Weight programming routes to a specific bank via `wr_bank` (an
// explicit input, rather than derived from wr_col by division --
// keeps the addressing simple and avoids relying on runtime
// division/modulo semantics). Real hardware would decode this from
// a global column address in the memory controller; that's outside
// this module's scope.
//
// TOTAL_COLS = NUM_BANKS * COLS_PER_BANK. For the real Metis spec:
// ROWS=512, COLS_PER_BANK=32, NUM_BANKS=16 -> TOTAL_COLS=512.
// Default here is reduced for simulation speed (matches imc_bank's
// Phase A defaults): ROWS=16, COLS_PER_BANK=4, NUM_BANKS=4.
//
// Reference: Metis AIPU ISSCC 2024, Fig 11.3.2/11.3.4 -- 16 banks
// tiling to the full 512x512 crossbar.
// ============================================================

module mvm_engine #(
    parameter int ROWS             = 16,
    parameter int COLS_PER_BANK    = 4,
    parameter int NUM_BANKS        = 4,
    parameter int WEIGHT_SETS      = 4,
    parameter int WBITS            = 8,
    parameter int ABITS            = 8,
    parameter int BITSERIAL_DEPTH  = ABITS,
    parameter int ACC_WIDTH        = 26,
    parameter int TOTAL_COLS       = NUM_BANKS * COLS_PER_BANK
)(
    input  logic                              clk,
    input  logic                              rst_n,

    // Weight programming interface -- wr_bank selects which bank's
    // weight_cell_array receives the write; wr_col is LOCAL to that
    // bank (0..COLS_PER_BANK-1), not a global column index.
    input  logic                              wr_en,
    input  logic [$clog2(NUM_BANKS)-1:0]      wr_bank,
    input  logic [$clog2(ROWS)-1:0]           wr_row,
    input  logic [$clog2(COLS_PER_BANK)-1:0]  wr_col,
    input  logic [$clog2(WEIGHT_SETS)-1:0]    wr_set,
    input  logic signed [WBITS-1:0]           wr_data,

    // Compute interface -- broadcast to all banks (shared activation
    // rows, as in the physical crossbar)
    input  logic                              start,
    input  logic [$clog2(WEIGHT_SETS)-1:0]    active_set,
    input  logic [ROWS*ABITS-1:0]             act_vec_flat,

    // result_flat: bank b's columns occupy
    //   [(b+1)*COLS_PER_BANK*ACC_WIDTH-1 -: COLS_PER_BANK*ACC_WIDTH]
    // i.e. global column index = b*COLS_PER_BANK + local_col
    output logic [TOTAL_COLS*ACC_WIDTH-1:0]   result_flat,
    output logic                              done,
    output logic                              busy
);

    // Per-bank write-enable decode: only the addressed bank sees
    // wr_en=1. wr_bank comparison against genvar `gbank` is a
    // constant-vs-variable equality check -- safe (not an indexed
    // part-select or bit-select, just a comparison).
    logic [NUM_BANKS-1:0] bank_wr_en;
    logic done_arr [NUM_BANKS];
    logic busy_arr [NUM_BANKS];

    genvar gbank;
    generate
        for (gbank = 0; gbank < NUM_BANKS; gbank++) begin : g_bank

            assign bank_wr_en[gbank] = wr_en && (wr_bank == gbank);

            logic [COLS_PER_BANK*ACC_WIDTH-1:0] bank_result;

            imc_bank #(
                .ROWS(ROWS), .COLS(COLS_PER_BANK), .WEIGHT_SETS(WEIGHT_SETS),
                .WBITS(WBITS), .ABITS(ABITS), .BITSERIAL_DEPTH(BITSERIAL_DEPTH),
                .ACC_WIDTH(ACC_WIDTH)
            ) u_bank (
                .clk(clk), .rst_n(rst_n),
                .wr_en(bank_wr_en[gbank]), .wr_row(wr_row), .wr_col(wr_col),
                .wr_set(wr_set), .wr_data(wr_data),
                .start(start), .active_set(active_set), .act_vec_flat(act_vec_flat),
                .result_flat(bank_result),
                .done(done_arr[gbank]), .busy(busy_arr[gbank])
            );

            // Place this bank's result into its slice of the global
            // result_flat -- gbank is a genvar (compile-time
            // constant per generated instance), so this part-select
            // is constant-indexed and safe.
            assign result_flat[(gbank+1)*COLS_PER_BANK*ACC_WIDTH-1 -: COLS_PER_BANK*ACC_WIDTH] = bank_result;

        end
    endgenerate

    // All banks share identical timing (same ROWS, same
    // BITSERIAL_DEPTH, same start pulse) -- bank 0's done/busy is
    // representative of all banks.
    assign done = done_arr[0];
    assign busy = busy_arr[0];

endmodule
