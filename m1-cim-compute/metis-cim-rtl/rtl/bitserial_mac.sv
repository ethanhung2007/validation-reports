// ============================================================
// bitserial_mac.sv
// Per-column bit-serial multiply-accumulate tree.
//
// Physical picture: every bit-cycle, ALL ROWS activation bits arrive
// in parallel (that's the crossbar's parallelism). For each column,
// this module multiplies each row's activation bit against that
// row's stationary weight (1-bit x WBITS -> WBITS partial product),
// sums the ROWS partial products with an adder tree, shifts the sum
// left by the current bit index, and accumulates into a running
// column result. The final activation bit is the two's-complement
// sign bit, so its shifted term is subtracted. After
// BITSERIAL_DEPTH cycles the accumulator holds the full signed INT8
// x INT8 dot-product result for that column and holds steady
// (accumulation gated by `en`, not free-running).
//
// COL_IDX (compile-time, via generate) selects which column of the
// flattened weight matrix this instance reads. `weights_flat` is a
// PACKED vector (row-major, (r*COLS+c)*WBITS +: WBITS) rather than
// an unpacked array -- see weight_cell_array.sv for why.
//
// Reference: Metis AIPU ISSCC 2024 -- "accumulation over 8 input bit
// cycles", 26b accumulators per bank (Fig 11.3.4a).
// ============================================================

module bitserial_mac #(
    parameter int ROWS             = 16,
    parameter int COLS             = 4,
    parameter int COL_IDX          = 0,
    parameter int WBITS            = 8,
    parameter int BITSERIAL_DEPTH  = 8,
    parameter int ACC_WIDTH        = 26
)(
    input  logic                              clk,
    input  logic                              rst_n,
    input  logic                              clear,
    input  logic                              en,

    input  logic [ROWS-1:0]                    act_bits,
    input  logic [ROWS*COLS*WBITS-1:0]         weights_flat,

    output logic signed [ACC_WIDTH-1:0]        acc,
    output logic                               valid
);

    logic [$clog2(BITSERIAL_DEPTH+1)-1:0] bit_idx;

    // Extract this column's ROWS weights from the flattened matrix.
    // gr is a compile-time genvar; the (r*COLS+COL_IDX) index resolves
    // to a constant per generated instance, so this is a plain
    // constant part-select -- fully supported.
    logic signed [WBITS-1:0] w [ROWS];
    genvar gr;
    generate
        for (gr = 0; gr < ROWS; gr++) begin : g_w
            assign w[gr] = weights_flat[((gr*COLS+COL_IDX)+1)*WBITS-1 -: WBITS];
        end
    endgenerate

    logic signed [WBITS-1:0] partial [ROWS];
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            partial[r] = act_bits[r] ? w[r] : '0;
        end
    end

    logic signed [WBITS+$clog2(ROWS)-1:0] row_sum;
    always_comb begin
        row_sum = '0;
        for (int r = 0; r < ROWS; r++) begin
            row_sum += partial[r];
        end
    end

    logic signed [ACC_WIDTH-1:0] row_sum_ext;
    assign row_sum_ext = row_sum;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc     <= '0;
            bit_idx <= '0;
        end else if (clear) begin
            acc     <= '0;
            bit_idx <= '0;
        end else if (en && bit_idx < BITSERIAL_DEPTH) begin
            if (bit_idx == BITSERIAL_DEPTH - 1)
                acc <= acc - (row_sum_ext <<< bit_idx);
            else
                acc <= acc + (row_sum_ext <<< bit_idx);
            bit_idx <= bit_idx + 1'b1;
        end
    end

    assign valid = (bit_idx == BITSERIAL_DEPTH);

endmodule
