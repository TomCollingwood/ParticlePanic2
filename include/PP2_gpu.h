#ifndef _PP2GPU_H_
#define _PP2GPU_H_

#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <thrust/device_vector.h>
#include <vector>

// Needed for output functions within the kernel
#include <stdio.h>

namespace PP2_GPU {
    thrust::device_vector<float> d_Px;
    thrust::device_vector<float> d_Py;
    thrust::device_vector<float> d_Vx;
    thrust::device_vector<float> d_Vy;
    float * d_Px_ptr;
    float * d_Py_ptr;
    float * d_Vx_ptr;
    float * d_Vy_ptr;

    int m_numPoints = 0;
    int m_gridResolution = 4;

    thrust::device_vector<unsigned int> d_cellOcc;
    thrust::device_vector<unsigned int> d_hash;

    unsigned int * d_hash_ptr;
    unsigned int * d_cellOcc_ptr;

    //-------------------------- KERNELS ----------------------------

//    __global__ void pointHash2D(unsigned int *hash,
//                              const float *Px,
//                              const float *Py,
//                              //const float *Pz,
//                              const unsigned int N,
//                              const unsigned int res);

//    __global__ void countCellOccupancy(unsigned int *cellOcc,
//                                       unsigned int *hash,
//                                       unsigned int nCells,
//                                       unsigned int nPoints);

    void initData();

    void hashOccSort();

    void addParticle(float P_x, float P_y, float V_x, float V_y);
}

#endif //_RAND_GPU_H
