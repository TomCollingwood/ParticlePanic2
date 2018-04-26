/// \file World.h
/// \brief contains all particles and methods to draw and update them
/// \author Thomas Collingwood
/// \version 1.0
/// \date 26/4/16 Updated to NCCA Coding standard
/// Revision History : See https://github.com/TomCollingwood/ParticlePanic

#ifndef _WORLDGPU_H_
#define _WORLDGPU_H_

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

class ParticlesData;

/**
 * @brief The Scene class
 */
class WorldGPU
{
public:
    WorldGPU();

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief WorldGPU         Parameterized constructor
    /// \param[in] _num_points  Number of particles
    /// \param[in] _iRadius     Interaction radius of particles
    /// \param[in] _timestep    Timestep of simulation
    /// \param[in] _gridRes     Resolution of spatial hash
    //----------------------------------------------------------------------------------------------------------------------
    WorldGPU(uint _num_points, float _iRadius, float _timestep, uint _gridRes);

    ~WorldGPU();

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief simulate         Simulate the particles for a frame
    /// \param[in] _substeps    Number of substeps to simulate (each substep lasts a timestep)
    //----------------------------------------------------------------------------------------------------------------------
    void simulate(int _substeps);

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief getNumPoints Returns number of particles in system
    /// \return             Number of particles
    //----------------------------------------------------------------------------------------------------------------------
    int getNumPoints();

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief dumpToGeo    Exports particles positions to Houdini geo file
    /// \param[in] _cnt     Frame number
    //----------------------------------------------------------------------------------------------------------------------
    void dumpToGeo(const uint _cnt);

private: // data
    int m_numPoints = 100;
    int m_gridResolution = 4;
    float m_interactionradius = 0.05f;
    int nCells = m_gridResolution*m_gridResolution;
    float m_timestep = 0.02f;
    ParticlesData * m_particlesData;
    bool m_firstTime = true;


};

#endif // WORLD_H
