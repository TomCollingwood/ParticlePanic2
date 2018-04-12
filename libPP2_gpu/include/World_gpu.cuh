#ifndef _WORLDGPU_CUH_
#define _WORLDGPU_CUH_

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

//// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>


extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, size_t size);

    void hashOccSort(int _num_points,
                 int _gridRes,
                 unsigned int * _d_hash_ptr,
                 unsigned int * _d_cellOcc_ptr,
                 unsigned int * _d_scatterAdd_ptr,
                 float * _Px,
                 float * _Py,
                 float * _prevPx,
                 float * _prevPy,
                 float * _Vx,
                 float * _Vy);

    void viscosity(unsigned int _N,
                   unsigned int _gridRes,
                   float _iRadius,
                   float _timestep,
                   float *_P_x,
                   float *_P_y,
                   float *_V_x,
                   float *_V_y,
                   unsigned int *_d_hash,
                   unsigned int *_d_cellOcc,
                   unsigned int *_d_scatterAdd);

    void integrate(unsigned int _N,
                   float _timestep,
                   float * _P_x,
                   float * _P_y,
                   float * _prevP_x,
                   float * _prevP_y,
                   float * _V_x,
                   float * _V_y);

    void densityD(unsigned int _N,
                  unsigned int _gridRes,
                  float _iRadius,
                  float _timestep,
                  float * _P_x,
                  float * _P_y,
                  float * _V_x,
                  float * _V_y,
                  unsigned int * _d_hash,
                  unsigned int * _d_cellOcc,
                  unsigned int * _d_scatterAdd);

    void updateVelocity(unsigned int _N,
                        float _timestep,
                        float *_P_x,
                        float *_P_y,
                        float *_prevP_x,
                        float *_prevP_y,
                        float *_V_x,
                        float *_V_y);

    void addGravity(unsigned int _N,
                    float *_V_x,
                    float *_V_y);

}

#endif
