#include <math.h>
#include "/gem5/include/gem5/m5ops.h"
#define N 14336
static float gate[N], up[N], out[N];
static volatile float sink;

static void op(float *gate, float *up, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
    sink = out[0];
}
int main() {
    for (int i = 0; i < N; i++) { gate[i] = (float)i * 0.001f; up[i] = (float)(N-i) * 0.001f; }
    op(gate, up, out, N); op(gate, up, out, N);
    m5_reset_stats(0, 0);
    op(gate, up, out, N);
    m5_dump_stats(0, 0);
    return 0;
}
