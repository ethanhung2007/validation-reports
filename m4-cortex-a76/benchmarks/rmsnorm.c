#include <math.h>
#include "/gem5/include/gem5/m5ops.h"
#define N 4096
static float x[N], w[N], out[N];
static volatile float sink;

static void op(float *x, float *w, float *out, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float rms = 1.0f / sqrtf(sum / n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * rms * w[i];
    sink = out[0];
}
int main() {
    for (int i = 0; i < N; i++) { x[i] = (float)i * 0.001f; w[i] = 1.0f; }
    op(x, w, out, N); op(x, w, out, N);
    m5_reset_stats(0, 0);
    op(x, w, out, N);
    m5_dump_stats(0, 0);
    return 0;
}
