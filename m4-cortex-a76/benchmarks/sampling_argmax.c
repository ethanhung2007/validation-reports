#include "/gem5/include/gem5/m5ops.h"
#define N 128256
static float logits[N];
static volatile int result;

static int op(float *x, int n) {
    /* Break loop-carried dependency with 4-way unroll */
    float m0 = x[0], m1 = x[1], m2 = x[2], m3 = x[3];
    int   i0 = 0,    i1 = 1,    i2 = 2,    i3 = 3;
    for (int i = 4; i < n - 3; i += 4) {
        if (x[i]   > m0) { m0 = x[i];   i0 = i;   }
        if (x[i+1] > m1) { m1 = x[i+1]; i1 = i+1; }
        if (x[i+2] > m2) { m2 = x[i+2]; i2 = i+2; }
        if (x[i+3] > m3) { m3 = x[i+3]; i3 = i+3; }
    }
    if (m1 > m0) { m0 = m1; i0 = i1; }
    if (m2 > m0) { m0 = m2; i0 = i2; }
    if (m3 > m0) { i0 = i3; }
    return i0;
}
int main() {
    for (int i = 0; i < N; i++) logits[i] = (float)i * 0.0001f;
    result = op(logits, N);
    result = op(logits, N);
    m5_reset_stats(0, 0);
    result = op(logits, N);
    m5_dump_stats(0, 0);
    return 0;
}
