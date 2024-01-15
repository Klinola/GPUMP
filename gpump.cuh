#ifndef GPUMP_CUH
#define GPUMP_CUH
#include <cuda.h>
#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <stdio.h>

typedef struct {
    unsigned int n[50];
} fe_num_t;
__device__ int mpCompare(const fe_num_t* x, const fe_num_t* y);
__device__ void mpAdd(fe_num_t* result, const fe_num_t* x, const fe_num_t* y);
__device__ void mpSubtract(fe_num_t* result, const fe_num_t* x, const fe_num_t* y);
__device__ void mpModularAdd(fe_num_t* result, const fe_num_t* x, const fe_num_t* y, const fe_num_t* mod);
__device__ void mpMul(fe_num_t* result, const fe_num_t* x, const fe_num_t* y);
__device__ void barrettReduction(fe_num_t* r, const fe_num_t* z, const fe_num_t* p, const fe_num_t* mu, int k);
#endif