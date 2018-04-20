/// \file World.h
/// \brief contains all particles and methods to draw and update them
/// \author Thomas Collingwood
/// \version 1.0
/// \date 26/4/16 Updated to NCCA Coding standard
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
 * @brief The Scene class
 */
class WorldCPU
{
public:
    /// A constructor, called when this class is instanced in the form of an object
    WorldCPU();

    /// A virtual destructor, in case we want to inherit from this class
    ~WorldCPU();

    //----------------------------------------------------------------------------------------------------------------------
    /// \brief update                   updates the particles in the world according to SPH algorithms. Called in timer.
    /// \param[out] o_updateinprogress  bool that is set when update is in progress
    //----------------------------------------------------------------------------------------------------------------------
    void simulate();

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

    void dumpToObj(const uint cnt);

    void dumpToGeo(const uint cnt);

    void pointHash();


private:
    /// Keep track of whether this has been initialised - otherwise it won't be ready to draw!
    bool m_isInit;

    // PARTICLES
    std::vector<Particle> m_particles;
    std::vector<ParticleProperties> m_particleTypes;

    // SPATIAL HASH
    std::vector<std::vector<Particle *>> m_grid;

    // WORLD SIZE ATTRIBUTES


    //---------------------------NEEDED---------------------------------------------
    int m_gridResolution = 4;
    int m_num_points = 10000;
    float m_interactionradius = 0.005f;
    bool m_gravity = true;
    double m_timestep=0.02f;

};

#endif // WORLD_H
