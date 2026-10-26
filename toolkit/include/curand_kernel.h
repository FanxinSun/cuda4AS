#ifndef CURAND_KERNEL_H
#define CURAND_KERNEL_H

#include <stdint.h>
#include "host_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef QUALIFIERS
#define QUALIFIERS static inline __attribute__((always_inline))
#endif

typedef struct curandStateXORWOW {
    unsigned int d;
    unsigned int v[5];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
} curandStateXORWOW_t;

typedef struct curandStateXORWOW curandState_t;

QUALIFIERS void curand_init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandState_t *state) {
    state->d = (unsigned int)(seed ^ subsequence ^ offset);
    state->v[0] = 123456789;
    state->v[1] = 362436069;
    state->v[2] = 521288629;
    state->v[3] = 88675123;
    state->v[4] = 5783321;
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
}

QUALIFIERS unsigned int curand(curandState_t *state) {
    unsigned int t = state->v[0] ^ (state->v[0] >> 2);
    state->v[0] = state->v[1];
    state->v[1] = state->v[2];
    state->v[2] = state->v[3];
    state->v[3] = state->v[4];
    state->v[4] = (state->v[4] ^ (state->v[4] << 4)) ^ (t ^ (t << 1));
    state->d += 362437;
    return state->v[4] + state->d;
}

QUALIFIERS float curand_uniform(curandState_t *state) {
    return (float)curand(state) / 4294967296.0f;
}

QUALIFIERS double curand_uniform_double(curandState_t *state) {
    return (double)curand(state) / 4294967296.0;
}

#ifdef __cplusplus
}
#endif

#endif // CURAND_KERNEL_H
