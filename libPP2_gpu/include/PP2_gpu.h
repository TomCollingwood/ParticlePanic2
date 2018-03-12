
#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

// Needed for output functions within the kernel
#include <stdio.h>

namespace Rand_GPU {
    thrust::device_vector<float> d_Px;
    thrust::device_vector<float> d_Py;
    float * d_Px_ptr;
    float * d_Py_ptr;

    thrust::device_vector<unsigned int> d_cellOcc;
    thrust::device_vector<unsigned int> d_hash(NUM_POINTS);

}

#endif //_RAND_GPU_H
