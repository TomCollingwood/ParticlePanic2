#ifndef _PARTICLES_DATA_CUH_
#define _PARTICLES_DATA_CUH_

#include <thrust/device_vector.h>

class ParticlesData
{
public:
    ParticlesData() = default;
    ~ParticlesData();

    thrust::device_vector<float> d_Px;
    thrust::device_vector<float> d_Py;
    thrust::device_vector<float> d_prevPx;
    thrust::device_vector<float> d_prevPy;
    thrust::device_vector<float> d_Vx;
    thrust::device_vector<float> d_Vy;
    thrust::device_vector<unsigned int> d_hash;
    thrust::device_vector<unsigned int> d_cellOcc;
    thrust::device_vector<unsigned int> d_scatterAdd;
};

#endif
