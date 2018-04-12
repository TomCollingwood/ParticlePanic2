#include <stdio.h>
#include <time.h>
#include <iostream>
#include <math.h>

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

#include "particles_kernel_impl.cuh"

extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

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
                 float * _Vy
                 )
    {
        if(_num_points==0) return;
        unsigned int nThreads = 1024;
        unsigned int nBlocks = _num_points / nThreads + 1;
        unsigned int nCells = _gridRes*_gridRes;

        castPointers();

        pointHash2D<<<nBlocks, nThreads>>>(_d_hash_ptr,
                                           _Px,
                                           _Py,
                                           _num_points,
                                           _gridRes);

        cudaThreadSynchronize();

        thrust::device_vector<float> d_Px(_Px, _Px + _num_points);
        thrust::device_vector<float> d_Py(_Py, _Py + _num_points);
        thrust::device_vector<float> d_prevPx(_prevPx, _prevPx + _num_points);
        thrust::device_vector<float> d_prevPy(_prevPy, _prevPy + _num_points);
        thrust::device_vector<float> d_Vx(_Vx, _Vx + _num_points);
        thrust::device_vector<float> d_Vy(_Vy, _Vy + _num_points);
        thrust::device_vector<unsigned int> d_hash(_d_hash_ptr, _d_hash_ptr+_num_points);
        thrust::device_vector<unsigned int> d_cellOcc(_d_cellOcc_ptr, _d_cellOcc_ptr+nCells);
        thrust::device_vector<unsigned int> d_scatterAdd(_d_scatterAdd_ptr, _d_scatterAdd_ptr+nCells);

        auto tuple = thrust::make_tuple( d_Px.begin(), d_Py.begin(), d_Vx.begin(), d_Vy.begin(), d_prevPx.begin(), d_prevPy.begin());
        auto zippy = thrust::make_zip_iterator(tuple);
        thrust::sort_by_key(d_hash.begin(), d_hash.end(), zippy);

        cudaThreadSynchronize();

        thrust::exclusive_scan(d_cellOcc.begin(),d_cellOcc.end(),d_scatterAdd.begin());

        cudaThreadSynchronize();

        countCellOccupancy<<<nBlocks, nThreads>>>(_d_cellOcc_ptr, _d_hash_ptr, nCells, _num_points);

        cudaThreadSynchronize();
    }

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
                   unsigned int *_d_scatterAdd)
    {

        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;

        viscosityD<<<nBlocks,nThreads>>>(_N,
                                        _gridRes,
                                        _iRadius,
                                        _timestep,
                                        _P_x,
                                        _P_y,
                                        _V_x,
                                        _V_y,
                                        _hash,
                                        _cellOcc,
                                        _scatterAdd);
    }

    void integrate(unsigned int _N,
                   float _timestep,
                   float * _P_x,
                   float * _P_y,
                   float * _prevP_x,
                   float * _prevP_y,
                   float * _V_x,
                   float * _V_y)
    {
        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;

        integrateD<<<nBlocks,nThreads>>>(_N,
                                        _timestep,
                                        _P_x,
                                        _P_y,
                                        _prevP_x,
                                        _prevP_y,
                                        _V_x,
                                        _V_y);
    }

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
                  unsigned int * _d_scatterAdd)
    {

        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;

        densityD<<<nBlocks,nThreads>>>(_N,
                                      _gridRes,
                                      _iRadius,
                                      _timestep,
                                      _P_x,
                                      _P_y,
                                      _V_x,
                                      _V_y,
                                      _d_hash,
                                      _d_cellOcc,
                                      _d_scatterAdd);
    }

    void updateVelocity(unsigned int _N,
                        float _timestep,
                        float *_P_x,
                        float *_P_y,
                        float *_prevP_x,
                        float *_prevP_y,
                        float *_V_x,
                        float *_V_y)
    {
        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;
        updateVelocityD<<<nBlocks,nThreads>>>(_N,
                                               _timestep,
                                               _P_x,
                                               _P_y,
                                               _prevP_x,
                                               _prevP_y,
                                               _V_x,
                                               _V_y);
    }

    void addGravity(unsigned int _N,
                    float *_V_x,
                    float *_V_y)
    {
        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;
        addGravityD<<<nBlocks,nThreads>>>(_N,_V_x,_V_y);
    }
}

