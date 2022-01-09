#include "dbg.h"

float *dthf(float *data, unsigned int len) {
    auto l = len * sizeof(float);
    void *p = malloc(l);
    cudaMemcpy(p, data, l, cudaMemcpyDeviceToHost);
    return (float *) p;
}

unsigned int *dthu(unsigned int *data, unsigned int len) {
    auto l = len * sizeof(unsigned int);
    void *p = malloc(l);
    cudaMemcpy(p, data, l, cudaMemcpyDeviceToHost);
    return (unsigned int *) p;
}

pf1000 ddf(float *data, unsigned int len) {
    return (pf1000) dthf(data, len);
}

pu1000 ddu(unsigned int *data, unsigned int len) {
    return (pu1000) dthu(data, len);
}
