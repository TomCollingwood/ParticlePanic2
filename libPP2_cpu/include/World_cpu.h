/// \file World.h
/// \brief contains all particles and methods to draw and update them
/// \author Thomas Collingwood
/// \version 2.0
/// \date 23/4/18 Updated to NCCA Coding standard
/// Revision History : See https://github.com/TomCollingwood/ParticlePanic

#ifndef _WORLDCPU_H_
#define _WORLDCPU_H_

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstdlib>

#include "Vec3_cpu.h"
#include "Particle_cpu.h"
#include "ParticleProperties_cpu.h"


/**
 * @brief The Wo class
 */
class WorldCPU
{
public:
    WorldCPU();

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief WorldCPU     Parameterized constructor
    /// \param _num_points  Number of particles to initialize
    /// \param _iRadius     The radius of interaction between particles
    /// \param _timestep    The timestep
    /// \param _gridRes     The resolution of the spatial hash
    //----------------------------------------------------------------------------------------------------------------------
    WorldCPU(int _num_points, float _iRadius, float _timestep, int _gridRes);

    ~WorldCPU();

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief update                   updates the particles in the world according to SPH algorithms. Called in timer.
    /// \param[out] o_updateinprogress  bool that is set when update is in progress
    //----------------------------------------------------------------------------------------------------------------------
    void simulate(int _substeps);

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief getSurroundingParticles  gets all particles in surrounding grids.
    /// \param[in] _thiscell            the centre cell in which to search for surrounding particles from
    /// \param[in] _numsur              how far out neightbouring particles can be (counted in grid squares)
    ///                                 if numsur==1 then it would return all particles in 3x3 grid around thiscell
    /// \param[in] _withwalls           if true will also include particles that are of wall type
    /// \return                         returns vector of pointers to the surrounding particles
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<Particle *> getSurroundingParticles(int thiscell,int numsur, bool withwalls) const;

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief hashParticles  takes m_grid and m_particles, using spatial hash organises the particles into
    ///                       buckets / squares as pointers
    //----------------------------------------------------------------------------------------------------------------------
    void hashParticles();

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief dumpToGeo    Creates Houdini Geo file with the simulated particles
    /// \param cnt[in]      What frame to save the filename as
    //----------------------------------------------------------------------------------------------------------------------
    void dumpToGeo(const uint cnt);

private:
    //----------------------------------------------------------------------------------------------------------------------
    /// \brief initData private function that is called in the constructors sets particles in
    ///                 dambreaker position
    //----------------------------------------------------------------------------------------------------------------------
    void initData();

    std::vector<Particle> m_particles;
    std::vector<ParticleProperties> m_particleTypes;

    // SPATIAL HASH
    //----------------------------------------------------------------------------------------------------------------------
    /// \brief m_grid   Contains particles ordered according to position
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<std::vector<Particle *>> m_grid;

    int m_gridResolution = 4;
    int m_num_points = 100;
    float m_interactionradius = 0.05f;
    bool m_gravity = true;
    double m_timestep=0.02f;
};

#endif // WORLD_H
