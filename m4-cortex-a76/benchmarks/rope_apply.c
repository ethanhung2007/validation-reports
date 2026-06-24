#include <math.h>
#include "/gem5/include/gem5/m5ops.h"
#define N 4096
static float x[N], out[N];
static volatile float sink;

static void op(float *x, float *out, int n) {
    for (int i = 0; i < n; i += 2) {
        float cos_v = cosf((float)(i/2) * 0.0001f);
        float sin_v = sinf((float)(i/2) * 0.0001f);
        out[i]   = x[i] * cos_v - x[i+1] * sin_v;
        out[i+1] = x[i] * sin_v + x[i+1] * cos_v;
    }
    sink = out[0];
}
int main() {
    for (int i = 0; i < N; i++) x[i] = (float)i * 0.001f;
    op(x, out, N); op(x, out, N);
    m5_reset_stats(0, 0);
    op(x, out, N);
    m5_dump_stats(0, 0);
    return 0;
}
