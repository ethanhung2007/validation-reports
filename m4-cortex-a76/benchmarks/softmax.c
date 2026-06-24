#include <math.h>
#include "/gem5/include/gem5/m5ops.h"
#ifndef KV
#define KV 512
#endif
static float x[KV], out[KV];
static volatile float sink;

static void op(float *x, float *out, int n) {
    float max_v = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_v) max_v = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(x[i] - max_v); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
    sink = out[0];
}
int main() {
    for (int i = 0; i < KV; i++) x[i] = (float)i * 0.01f;
    op(x, out, KV); op(x, out, KV);
    m5_reset_stats(0, 0);
    op(x, out, KV);
    m5_dump_stats(0, 0);
    return 0;
}
