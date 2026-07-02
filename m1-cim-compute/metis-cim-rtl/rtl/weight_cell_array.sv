
// ROWS x COLS x WEIGHT_SETS weight storage for one IMC bank.
//
// IMPLEMENTATION NOTE: the read port is a single flattened PACKED
// vector (rd_flat), not an unpacked array-of-vectors. Icarus Verilog
// (and some other tools) do not reliably propagate values through
// unpacked-array module ports -- packed vectors are the portable,
// synthesis-safe choice. Bit position for weight(r,c) is
// (r*COLS+c)*WBITS +: WBITS  (row-major).
//
// Reference: Metis AIPU ISSCC 2024, Fig 11.3.4(a)/(b) -- 4 weight
// sets per cell, bit-parallel-to-bit-serial converter at the input.

module weight_cell_array #(
    parameter int ROWS         = 16,
    parameter int COLS         = 4,
    parameter int WEIGHT_SETS  = 4,
    parameter int WBITS        = 8
)(
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic                              wr_en,
    input  logic [$clog2(ROWS)-1:0]           wr_row,
    input  logic [$clog2(COLS)-1:0]           wr_col,
    input  logic [$clog2(WEIGHT_SETS)-1:0]    wr_set,
    input  logic signed [WBITS-1:0]           wr_data,

    input  logic [$clog2(WEIGHT_SETS)-1:0]    rd_set,
    output logic [ROWS*COLS*WBITS-1:0]        rd_flat   // packed; unpack via (r*COLS+c)*WBITS +: WBITS
);

    logic signed [WBITS-1:0] mem [ROWS][COLS][WEIGHT_SETS];

    always_ff @(posedge clk) begin
        if (wr_en) begin
            mem[wr_row][wr_col][wr_set] <= wr_data;
        end
    end

    genvar gr, gc;
    generate
        for (gr = 0; gr < ROWS; gr++) begin : g_row
            for (gc = 0; gc < COLS; gc++) begin : g_col
                assign rd_flat[((gr*COLS+gc)+1)*WBITS-1 -: WBITS] = mem[gr][gc][rd_set];
            end
        end
    endgenerate

endmodule
