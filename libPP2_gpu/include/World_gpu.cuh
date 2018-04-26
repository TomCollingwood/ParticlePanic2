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

//----------------------------------------------------------------------------------------------------------------------
/// \brief initializeParticlesData  Initializes particles in dambreaker formation
/// \param[in] _num_points          Number of particles to initialize
/// \param[in] _gridRes             The resolution of the spatial hash
/// \return                         The ParticlesData object containing the particle data
//----------------------------------------------------------------------------------------------------------------------
ParticlesData * initializeParticlesData(const int _num_points, const int _gridRes);

//----------------------------------------------------------------------------------------------------------------------
/// \brief deleteData       Delete memory where the particle data is stored
/// \param[in,out] io_data  The data to delete
//----------------------------------------------------------------------------------------------------------------------
void deleteData(ParticlesData * io_data);

//----------------------------------------------------------------------------------------------------------------------
/// \brief dumpToGeoCUDA    Exports particle positions to Houdini Geo file
/// \param[in] _data        The ParticlesData object to export
/// \param[in] _cnt         The frame number
//----------------------------------------------------------------------------------------------------------------------
void dumpToGeoCUDA(ParticlesData *_data,
               const uint _cnt);

//----------------------------------------------------------------------------------------------------------------------
/// \brief simulateD        Runs all CUDA simulation steps in this function
/// \param[in] _N           Number of particles
/// \param[in] _gridRes     Resolution of spatial hash
/// \param[in] _iRadius     Interaction radius of particles
/// \param[in] _timestep    Timestep
/// \param[in,out] io_data  The particle data to use for simulation
//----------------------------------------------------------------------------------------------------------------------
void simulateD(unsigned int _N,
               unsigned int _gridRes,
               float _iRadius,
               float _timestep,
               ParticlesData * io_data);

//----------------------------------------------------------------------------------------------------------------------
/// \brief hashOccSort      Finds hash for each particle, sorts according to hash, counts cell occupancy
///                         and then performs exclusive scan on cell occupancy to get cell indexes (d_scatterAdd)
/// \param[in] _num_points  Number of particles
/// \param[in] _gridRes     Resolution of the spatial hash
/// \param[in,out] io_data  The particle data to use
//----------------------------------------------------------------------------------------------------------------------
void hashOccSort(int _num_points,
             int _gridRes,
             ParticlesData * io_data);

//----------------------------------------------------------------------------------------------------------------------
/// \brief viscosity        Calculates viscosity velocity additions to particles
/// \param[in] _N           Number of particles
/// \param[in] _gridRes     Resolution of spatial hash
/// \param[in] _iRadius     Interaction radius of particles
/// \param[in] _timestep    Timestep used in simulation
/// \param[in,out] io_data  The particle data to use for simulation
//----------------------------------------------------------------------------------------------------------------------
void viscosity(unsigned int _N,
               unsigned int _gridRes,
               float _iRadius,
               float _timestep,
               ParticlesData * io_data);

//----------------------------------------------------------------------------------------------------------------------
/// \brief integrate        Integrates the position with velocity
/// \param[in] _N           Number of particles
/// \param[in] _timestep    Timestep used in simulation
/// \param[in,out] io_data  The particle data to use for simulation
//----------------------------------------------------------------------------------------------------------------------
void integrate(unsigned int _N,
               float _timestep,
               ParticlesData * io_data);

//----------------------------------------------------------------------------------------------------------------------
/// \brief density          Calculates density for each particle and moves particle and surrounding particles accordingly.
/// \param[in] _N           Number of particles
/// \param[in] _gridRes     Resolution of spatial hash
/// \param[in] _iRadius     Interaction radius of the particles
/// \param[in] _timestep    Timestep
/// \param[in,out] io_data  The particle data to use for simulation
//----------------------------------------------------------------------------------------------------------------------
void density(unsigned int _N,
              unsigned int _gridRes,
              float _iRadius,
              float _timestep,
              ParticlesData * io_data);

//----------------------------------------------------------------------------------------------------------------------
/// \brief updateVelocity   Updates velocity based on previous and current position
/// \param[in] _N           Number of particles
/// \param[in] _timestep    Timestep used in simulation
/// \param[in,out] io_data  The particle data to use for simulation
//----------------------------------------------------------------------------------------------------------------------
void updateVelocity(unsigned int _N,
                    float _timestep,
                    ParticlesData * io_data);

//----------------------------------------------------------------------------------------------------------------------
/// \brief addGravity       Adds gravity acceleration to the velocity
/// \param[in] _N           Number of particles
/// \param[in,out] io_data  The particle data to use for simulation
//----------------------------------------------------------------------------------------------------------------------
void addGravity(unsigned int _N,
                ParticlesData * io_data);

#endif
