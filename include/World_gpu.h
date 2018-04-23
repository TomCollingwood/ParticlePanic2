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
    /// A constructor, called when this class is instanced in the form of an object
    WorldGPU();
    WorldGPU(uint _num_points, float _iRadius, float _timestep, uint _gridRes);

    /// A virtual destructor, in case we want to inherit from this class
    ~WorldGPU();

    void simulate(int _substeps);

    int getNumPoints();

    void dumpToGeo(const uint cnt);

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
