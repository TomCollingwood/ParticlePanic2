///
///  @file World.cpp
///  @brief contains all particles and methods to draw and update them

#include "include/World_gpu.h"
#include "include/World_gpu.cuh"
#include "include/particles_kernel.cuh"

WorldGPU::WorldGPU()
{
    m_particlesData = initializeParticlesData(m_numPoints,m_gridResolution);
}

WorldGPU::~WorldGPU()
{

}


void WorldGPU::simulate()
{
    if(m_firstTime)
    {
        hashOccSort(m_numPoints,
                    m_gridResolution,
                    m_particlesData);
        m_firstTime=false;
    }
    addGravity(m_numPoints,m_particlesData);

    viscosity(m_numPoints,
              m_gridResolution,
              m_interactionradius,
              m_timestep,
              m_particlesData);


    hashOccSort(m_numPoints,
                m_gridResolution,
                m_particlesData);

    integrate(m_numPoints,
              m_timestep,
              m_particlesData);

    density(m_numPoints,
            m_gridResolution,
            m_interactionradius,
            m_timestep,
            m_particlesData);

    updateVelocity(m_numPoints,
                   m_timestep,
                   m_particlesData);

}

void WorldGPU::dumpToGeo(const uint cnt)
{
    dumpToGeoCUDA(m_particlesData,cnt);
}
