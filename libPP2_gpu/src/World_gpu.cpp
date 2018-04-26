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

WorldGPU::WorldGPU(uint _num_points, float _iRadius, float _timestep, uint _gridRes)
{
    m_numPoints=_num_points;
    m_interactionradius=_iRadius;
    m_timestep=_timestep;
    m_gridResolution = _gridRes;
    m_particlesData = initializeParticlesData(m_numPoints,m_gridResolution);
}

WorldGPU::~WorldGPU()
{
    deleteData(m_particlesData);
}

void WorldGPU::simulate(int _substeps)
{
    for(int i = 0; i<_substeps; ++i)
    {
        if(m_firstTime)
        {
            hashOccSort(m_numPoints,
                        m_gridResolution,
                        m_particlesData);
            m_firstTime=false;
        }

       // simulateD(m_numPoints,m_gridResolution,m_interactionradius,m_timestep,m_particlesData);
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
}

void WorldGPU::dumpToGeo(const uint _cnt)
{
    dumpToGeoCUDA(m_particlesData,_cnt);
}
