#include "/gem5/include/gem5/m5ops.h"
#define N 4096
static float a[N], b[N];
static volatile float sink;

static void op(float *a, float *b, int n) {
    float c[N];
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
    sink = c[0];
}
int main() {
    for (int i = 0; i < N; i++) { a[i] = (float)i * 0.001f; b[i] = (float)(N-i) * 0.001f; }
    op(a, b, N); op(a, b, N);
    m5_reset_stats(0, 0);
    op(a, b, N);
    m5_dump_stats(0, 0);
    return 0;
}
