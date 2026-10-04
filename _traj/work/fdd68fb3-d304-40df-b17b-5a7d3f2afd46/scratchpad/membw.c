// STREAM-style CPU memory bandwidth: read (sum), write (fill), copy, triad; best of REPS.
// Bytes counted as STREAM does (explicit loads + stores; a store's read-for-ownership is not counted).
// gcc -O3 -march=native -fopenmp membw.c -o membw ; OMP_NUM_THREADS=N OMP_PLACES=... OMP_PROC_BIND=close ./membw
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N (256L * 1024 * 1024)   // doubles per array: 2 GiB
#define REPS 5

static double now(void) { return omp_get_wtime(); }

int main(void) {
    double *a = aligned_alloc(64, N * sizeof(double)), *b = aligned_alloc(64, N * sizeof(double)),
           *c = aligned_alloc(64, N * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
#pragma omp parallel for schedule(static)          // first touch by the same threads that measure
    for (long i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
    double best[6] = {1e9, 1e9, 1e9, 1e9, 1e9, 1e9}, sink = 0;
    for (int r = 0; r < REPS; r++) {
        double t = now(), s = 0;
#pragma omp parallel for reduction(+:s) schedule(static)
        for (long i = 0; i < N; i++) s += a[i];
        t = now() - t; if (t < best[0]) best[0] = t; sink += s;
        t = now();
#pragma omp parallel for schedule(static)
        for (long i = 0; i < N; i++) c[i] = 3.0;
        t = now() - t; if (t < best[1]) best[1] = t;
        t = now();
#pragma omp parallel for schedule(static)
        for (long i = 0; i < N; i++) c[i] = a[i];
        t = now() - t; if (t < best[2]) best[2] = t;
        t = now();
#pragma omp parallel for schedule(static)
        for (long i = 0; i < N; i++) a[i] = b[i] + 3.0 * c[i];
        t = now() - t; if (t < best[3]) best[3] = t;
        t = now();                                  // glibc memset/memcpy: non-temporal stores at this size
#pragma omp parallel
        { int k = omp_get_thread_num(), n = omp_get_num_threads(); long lo = N * k / n, hi = N * (k + 1) / n;
          memset(c + lo, 0, (hi - lo) * sizeof(double)); }
        t = now() - t; if (t < best[4]) best[4] = t;
        t = now();
#pragma omp parallel
        { int k = omp_get_thread_num(), n = omp_get_num_threads(); long lo = N * k / n, hi = N * (k + 1) / n;
          memcpy(c + lo, b + lo, (hi - lo) * sizeof(double)); }
        t = now() - t; if (t < best[5]) best[5] = t;
    }
    const double B = N * sizeof(double) / 1e9;   // GB per array pass
    printf("threads %2d  read %6.1f  write %6.1f  copy %6.1f  triad %6.1f | memset %6.1f  memcpy %6.1f  GB/s  (sink %g)\n",
           omp_get_max_threads(), B / best[0], B / best[1], 2 * B / best[2], 3 * B / best[3], B / best[4], 2 * B / best[5],
           sink > 0 ? 1.0 : 0.0);
    return 0;
}
