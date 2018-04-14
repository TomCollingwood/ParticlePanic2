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

#include "World_gpu.cuh"
#include "particles_kernel.cuh"
#include "particles_data.cuh"

extern "C"
{

//    void cudaInit(int argc, char **argv)
//    {
//        int devID;

//        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//        devID = findCudaDevice(argc, (const char **)argv);

//        if (devID < 0)
//        {
//            printf("No CUDA Capable devices found, exiting...\n");
//            exit(EXIT_SUCCESS);
//        }
//    }

    void allocateArray(void **devPtr, size_t size)
    {
        cudaMalloc(devPtr, size);
    }

    void initializeData(int _num_pointsX, int _num_pointsY, int _gridRes, ParticlesData ** _data)
    {
        (*_data) = new ParticlesData();

        int nCells = _gridRes*_gridRes;
        int nParticles = _num_pointsX*_num_pointsY;

        (*_data)->d_Px.resize(nParticles);
        (*_data)->d_Py.resize(nParticles);
        (*_data)->d_prevPx.resize(nParticles);
        (*_data)->d_prevPy.resize(nParticles);
        (*_data)->d_Vx.resize(nParticles,0.0f);
        (*_data)->d_Vy.resize(nParticles,0.0f);

        for(int x = 0; x<_num_pointsX; ++x)
        {
            for(int y=0; y<_num_pointsY; ++y)
            {
                (*_data)->d_Px[x*_num_pointsX + y]=0.0f+0.01f*x;
                (*_data)->d_Py[x*_num_pointsX + y]=0.0f+0.01f*y;
                (*_data)->d_prevPx[x*_num_pointsX + y]=0.0f+0.01f*x;
                (*_data)->d_prevPy[x*_num_pointsX + y]=0.0f+0.01f*y;
//                (*_data)->d_Vx[x*_num_pointsX + y]=0.0f;
//                (*_data)->d_Vy[x*_num_pointsX + y]=0.0f;
            }
        }
        (*_data)->d_hash.resize(nParticles,0);
        (*_data)->d_cellOcc.resize(nCells,0);
        (*_data)->d_scatterAdd.resize(nCells,0);
    }

    void hashOccSort(int _num_points,
                 int _gridRes,
                 ParticlesData * _data
                 )
    {
        if(_num_points==0) return;
        unsigned int nThreads = 1024;
        unsigned int nBlocks = _num_points / nThreads + 1;
        unsigned int nCells = _gridRes*_gridRes;

        pointHash2D<<<nBlocks, nThreads>>>(thrust::raw_pointer_cast(_data->d_hash.data()),
                                           thrust::raw_pointer_cast(_data->d_Px.data()),
                                           thrust::raw_pointer_cast(_data->d_Py.data()),
                                           _num_points,
                                           _gridRes);

        cudaThreadSynchronize();

        auto tuple = thrust::make_tuple( _data->d_Px.begin(), _data->d_Py.begin(), _data->d_Vx.begin(), _data->d_Vy.begin(), _data->d_prevPx.begin(), _data->d_prevPy.begin());
        auto zippy = thrust::make_zip_iterator(tuple);
        thrust::sort_by_key(_data->d_hash.begin(), _data->d_hash.end(), zippy); // bad alloc here

        cudaThreadSynchronize();

        thrust::exclusive_scan(_data->d_cellOcc.begin(),_data->d_cellOcc.end(),_data->d_scatterAdd.begin());

        cudaThreadSynchronize();

        countCellOccupancyD<<<nBlocks, nThreads>>>((uint *)thrust::raw_pointer_cast(_data->d_cellOcc.data()),
                                                   (uint *)thrust::raw_pointer_cast(_data->d_hash.data()),
                                                   nCells,
                                                   _num_points);

        cudaThreadSynchronize();
    }

    void viscosity(unsigned int _N,
                   unsigned int _gridRes,
                   float _iRadius,
                   float _timestep,
                   ParticlesData * _data)
    {

        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;

        viscosityD<<<nBlocks,nThreads>>>(_N,
                                        _gridRes,
                                        _iRadius,
                                        _timestep,
                                        (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                        (float *)thrust::raw_pointer_cast(_data->d_Py.data()),
                                        (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                        (float *)thrust::raw_pointer_cast(_data->d_Vy.data()),
                                        (uint *)thrust::raw_pointer_cast(_data->d_hash.data()),
                                        (uint *)thrust::raw_pointer_cast(_data->d_cellOcc.data()),
                                        (uint *)thrust::raw_pointer_cast(_data->d_scatterAdd.data()));
    }

    void integrate(unsigned int _N,
                   float _timestep,
                   ParticlesData * _data)
    {
        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;

        integrateD<<<nBlocks,nThreads>>>(_N,
                                        _timestep,
                                        (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                        (float *)thrust::raw_pointer_cast(_data->d_Py.data()),
                                        (float *)thrust::raw_pointer_cast(_data->d_prevPx.data()),
                                        (float *)thrust::raw_pointer_cast(_data->d_prevPy.data()),
                                        (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                        (float *)thrust::raw_pointer_cast(_data->d_Vy.data()));
    }

    void density(unsigned int _N,
                  unsigned int _gridRes,
                  float _iRadius,
                  float _timestep,
                  ParticlesData * _data)
    {

        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;

        densityD<<<nBlocks,nThreads>>>(_N,
                                      _gridRes,
                                      _iRadius,
                                      _timestep,
                                      (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                      (float *)thrust::raw_pointer_cast(_data->d_Py.data()),
                                      (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                      (float *)thrust::raw_pointer_cast(_data->d_Vy.data()),
                                      (uint *)thrust::raw_pointer_cast(_data->d_hash.data()),
                                      (uint *)thrust::raw_pointer_cast(_data->d_cellOcc.data()),
                                      (uint *)thrust::raw_pointer_cast(_data->d_scatterAdd.data()));
    }

    void updateVelocity(unsigned int _N,
                        float _timestep,
                        ParticlesData * _data)
    {
        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;
        updateVelocityD<<<nBlocks,nThreads>>>(_N,
                                              _timestep,
                                              (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                              (float *)thrust::raw_pointer_cast(_data->d_Py.data()),
                                              (float *)thrust::raw_pointer_cast(_data->d_prevPx.data()),
                                              (float *)thrust::raw_pointer_cast(_data->d_prevPy.data()),
                                              (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                              (float *)thrust::raw_pointer_cast(_data->d_Vy.data()));
    }

    void addGravity(unsigned int _N,
                    ParticlesData * _data)
    {
        unsigned int nThreads = 1024;
        unsigned int nBlocks = _N / nThreads + 1;
        addGravityD<<<nBlocks,nThreads>>>(_N,
                                          (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                          (float *)thrust::raw_pointer_cast(_data->d_Vy.data()));
    }
}

