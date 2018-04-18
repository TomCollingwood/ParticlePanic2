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

#include "particles_data.cuh"

extern "C"
{
//    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, size_t size);

    ParticlesData * initializeData(int _num_pointsX, int _num_pointsY, int _gridRes);

    void hashOccSort(int _num_points,
                 int _gridRes,
                 ParticlesData * _data);

    void pointHash(int _num_points,
                   int _gridRes,
                   ParticlesData * _data);

    void sortHash(ParticlesData * _data);

    void countCellOcc(int _num_points,
                   int _gridRes,
                   ParticlesData * _data);

    void exclusiveScan(ParticlesData * _data);

    void viscosity(unsigned int _N,
                   unsigned int _gridRes,
                   float _iRadius,
                   float _timestep,
                   ParticlesData * _data);

    void integrate(unsigned int _N,
                   float _timestep,
                   ParticlesData * _data);

    void density(unsigned int _N,
                  unsigned int _gridRes,
                  float _iRadius,
                  float _timestep,
                  ParticlesData * _data);

    void updateVelocity(unsigned int _N,
                        float _timestep,
                        ParticlesData * _data);

    void addGravity(unsigned int _N,
                    ParticlesData * _data);

}

#endif
