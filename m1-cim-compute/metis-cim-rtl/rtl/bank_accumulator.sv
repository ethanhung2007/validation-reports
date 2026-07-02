// ============================================================
// bank_accumulator.sv
// COLS parallel bit-serial MAC accumulators for one IMC bank.
//
// Physical picture: the bank has ROWS shared activation inputs
// (bit-serial) fanned out to COLS independent column compute paths.
// Each column reads its own ROWS weights (sliced from the flattened
// weight matrix) and accumulates its own dot product over
// BITSERIAL_DEPTH cycles. This is the "32 x 26-bit accumulators"
// stage from Fig 11.3.4(a).
//
// `result_flat` is a packed vector (ACC_WIDTH bits per column,
// column c at (c+1)*ACC_WIDTH-1 -: ACC_WIDTH) rather than an
// unpacked array -- see weight_cell_array.sv for why.
//
// Reference: Metis AIPU ISSCC 2024, MVM engine block diagram.
// ============================================================

module bank_accumulator #(
    parameter int ROWS             = 16,
    parameter int COLS             = 4,
    parameter int WBITS            = 8,
    parameter int BITSERIAL_DEPTH  = 8,
    parameter int ACC_WIDTH        = 26
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        clear,
    input  logic                        en,

    input  logic [ROWS-1:0]              act_bits,
    input  logic [ROWS*COLS*WBITS-1:0]   weights_flat,

    output logic [ACC_WIDTH*COLS-1:0]    result_flat,
    output logic                         valid
);

    logic valid_col [COLS];

    genvar gc;
    generate
        for (gc = 0; gc < COLS; gc++) begin : gen_col
            logic signed [ACC_WIDTH-1:0] acc_c;

            bitserial_mac #(
                .ROWS(ROWS), .COLS(COLS), .COL_IDX(gc),
                .WBITS(WBITS), .BITSERIAL_DEPTH(BITSERIAL_DEPTH),
                .ACC_WIDTH(ACC_WIDTH)
            ) u_mac (
                .clk(clk), .rst_n(rst_n), .clear(clear), .en(en),
                .act_bits(act_bits),
                .weights_flat(weights_flat),
                .acc(acc_c),
                .valid(valid_col[gc])
            );

            assign result_flat[(gc+1)*ACC_WIDTH-1 -: ACC_WIDTH] = acc_c;
        end
    endgenerate

    assign valid = valid_col[0]; // all columns share the same bit-cycle counter

endmodule
