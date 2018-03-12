#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <time.h>
#include <iostream>


//// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "PP2_gpu.h"

void PP2_GPU::initData()
{
    d_Px = thrust::device_vector<float>();
    d_Py = thrust::device_vector<float>();
    d_Px_ptr = thrust::raw_pointer_cast(&d_Px[0]);
    d_Py_ptr = thrust::raw_pointer_cast(&d_Py[0]);

    d_Vx = thrust::device_vector<float>();
    d_Vy = thrust::device_vector<float>();
    d_Vx_ptr = thrust::raw_pointer_cast(&d_Vx[0]);
    d_Vy_ptr = thrust::raw_pointer_cast(&d_Vy[0]);

    d_hash_ptr = thrust::raw_pointer_cast(&d_hash[0]);
    d_cellOcc_ptr = thrust::raw_pointer_cast(&d_cellOcc[0]);
}

//void PP2_GPU::hashOccSort()
//{
//    unsigned int nThreads = 1024;
//    unsigned int nBlocks = m_numPoints / nThreads + 1;

////    pointHash2D<<<nBlocks, nThreads>>>(d_hash_ptr, d_Px_ptr, d_Py_ptr,
////                                         m_numPoints,
////                                         m_gridResolution);

//    thrust::sort_by_key(d_hash.begin(), d_hash.end(),
//                            thrust::make_zip_iterator(
//                                thrust::make_tuple( d_Px.begin(), d_Py.begin(), d_Vx.begin(),d_Vy.begin())));
//}
