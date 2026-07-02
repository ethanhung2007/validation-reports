// ============================================================
// imc_bank.sv
// Single IMC bank: ROWS inputs x COLS outputs, bit-serial D-IMC MVM.
// This is the Phase A deliverable.
//
// Operation:
//   1. Host loads ROWS x COLS weight matrix via wr_* port (once,
//      amortized over many MAC operations -- "weight-stationary").
//   2. Host presents a full ROWS-wide activation vector (act_vec_flat,
//      packed: row r at (r+1)*ABITS-1 -: ABITS) and pulses `start`.
//   3. Bank internally serializes the activation vector bit-by-bit
//      (LSB first) over BITSERIAL_DEPTH cycles, feeding all ROWS bits
//      in parallel each cycle to bank_accumulator.
//   4. After BITSERIAL_DEPTH cycles, `done` asserts and result_flat
//      holds the dot product for each of the COLS outputs (packed,
//      column c at (c+1)*ACC_WIDTH-1 -: ACC_WIDTH).
//
// Ports are flattened PACKED vectors, not unpacked arrays -- see
// weight_cell_array.sv header for why (portability across sim tools).
//
// Reference: Metis AIPU ISSCC 2024, one of 16 banks tiling to the
// full 512x512 crossbar (Fig 11.3.2, 11.3.4).
// ============================================================

module imc_bank #(
    parameter int ROWS             = 16,
    parameter int COLS             = 4,
    parameter int WEIGHT_SETS      = 4,
    parameter int WBITS            = 8,
    parameter int ABITS            = 8,
    parameter int BITSERIAL_DEPTH  = ABITS,
    parameter int ACC_WIDTH        = 26
)(
    input  logic                              clk,
    input  logic                              rst_n,

    // Weight programming interface
    input  logic                              wr_en,
    input  logic [$clog2(ROWS)-1:0]           wr_row,
    input  logic [$clog2(COLS)-1:0]           wr_col,
    input  logic [$clog2(WEIGHT_SETS)-1:0]    wr_set,
    input  logic signed [WBITS-1:0]           wr_data,

    // Compute interface
    input  logic                              start,
    input  logic [$clog2(WEIGHT_SETS)-1:0]    active_set,
    input  logic [ROWS*ABITS-1:0]             act_vec_flat,

    output logic [ACC_WIDTH*COLS-1:0]         result_flat,
    output logic                              done,
    output logic                              busy
);

    // ── Weight storage ──────────────────────────────────────
    logic [ROWS*COLS*WBITS-1:0] weights_flat;

    weight_cell_array #(
        .ROWS(ROWS), .COLS(COLS), .WEIGHT_SETS(WEIGHT_SETS), .WBITS(WBITS)
    ) u_weights (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en), .wr_row(wr_row), .wr_col(wr_col), .wr_set(wr_set), .wr_data(wr_data),
        .rd_set(active_set),
        .rd_flat(weights_flat)
    );

    // ── Activation bit-serial feeder ────────────────────────
    // Unpack act_vec_flat into an internal array (safe -- internal,
    // not a port), latch it at `start`, then shift out one bit per
    // row per cycle (LSB first) for BITSERIAL_DEPTH cycles.
    logic signed [ABITS-1:0] act_latch [ROWS];
    logic [ROWS-1:0]         act_bits;
    logic [$clog2(BITSERIAL_DEPTH+1)-1:0] cycle_cnt;
    logic running;

    genvar gb;
    generate
        for (gb = 0; gb < ROWS; gb++) begin : g_actbit
            assign act_bits[gb] = act_latch[gb][0];
        end
    endgenerate

    // Unpack act_vec_flat -> act_vec_arr using variable SHIFT (not
    // variable part-select -- Icarus does not reliably support
    // variable-indexed part-selects `vec[(r+1)*W-1 -: W]` inside
    // procedural blocks, only constant/genvar-indexed ones).
    logic signed [ABITS-1:0] act_vec_arr [ROWS];
    logic [ABITS-1:0]        act_shift_tmp;
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            act_shift_tmp   = (act_vec_flat >> (r*ABITS)) & {ABITS{1'b1}};
            act_vec_arr[r]  = act_shift_tmp;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running   <= 1'b0;
            cycle_cnt <= '0;
            for (int r = 0; r < ROWS; r++) act_latch[r] <= '0;
        end else if (start && !running) begin
            for (int r = 0; r < ROWS; r++)
                act_latch[r] <= act_vec_arr[r];
            running   <= 1'b1;
            cycle_cnt <= '0;
        end else if (running) begin
            for (int r = 0; r < ROWS; r++) act_latch[r] <= act_latch[r] >>> 1;
            cycle_cnt <= cycle_cnt + 1'b1;
            if (cycle_cnt == BITSERIAL_DEPTH - 1) running <= 1'b0;
        end
    end

    assign busy = running;

    // ── Accumulation ─────────────────────────────────────────
    logic clear_acc;
    assign clear_acc = start && !running;

    bank_accumulator #(
        .ROWS(ROWS), .COLS(COLS), .WBITS(WBITS),
        .BITSERIAL_DEPTH(BITSERIAL_DEPTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_acc (
        .clk(clk), .rst_n(rst_n), .clear(clear_acc), .en(running),
        .act_bits(act_bits),
        .weights_flat(weights_flat),
        .result_flat(result_flat),
        .valid(done)
    );

endmodule
