#ifndef _PARTICLES_DATA_CUH_
#define _PARTICLES_DATA_CUH_

#include <thrust/device_vector.h>

class ParticlesData
{
public:
    ParticlesData() = default;
    ~ParticlesData() {}

    // Current posit
    thrust::device_vector<float> d_Px;
    thrust::device_vector<float> d_Py;

    // previous positions of particles
    thrust::device_vector<float> d_prevPx;
    thrust::device_vector<float> d_prevPy;

    // The velocities
    thrust::device_vector<float> d_Vx;
    thrust::device_vector<float> d_Vy;

    // list of hash values for particles
    thrust::device_vector<unsigned int> d_hash;
    // how many particles for each hash value
    thrust::device_vector<unsigned int> d_cellOcc;
    // particle indexes for each hash value
    thrust::device_vector<unsigned int> d_scatterAdd;
};

#endif
