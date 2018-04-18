#ifndef _PARTICLES_DATA_CUH_
#define _PARTICLES_DATA_CUH_

#include <thrust/device_vector.h>

class ParticlesData
{
public:
    ParticlesData() = default;
//    ParticlesData(int _numPoints, int _gridRes)
//    {
//        d_Px = thrust::device_vector<float>(_numPoints,0.0f);
//        d_Py = thrust::device_vector<float>(_numPoints,0.0f);
//        d_prevPx = thrust::device_vector<float>(_numPoints,0.0f);
//        d_prevPy = thrust::device_vector<float>(_numPoints,0.0f);
//        d_Vx = thrust::device_vector<float>(_numPoints,0.0f);
//        d_Vy = thrust::device_vector<float>(_numPoints,0.0f);
//        d_hash = thrust::device_vector<uint>(_numPoints,0);
//        d_cellOcc = thrust::device_vector<uint>(_gridRes*_gridRes,0);
//        d_scatterAdd = thrust::device_vector<uint>(_gridRes*_gridRes,0);
//    }

//    void sizeOut(int _numPoints, int _gridRes)
//    {
//        d_Px = thrust::device_vector<float>(_numPoints,0.0f);
//        d_Py = thrust::device_vector<float>(_numPoints,0.0f);
//        d_prevPx = thrust::device_vector<float>(_numPoints,0.0f);
//        d_prevPy = thrust::device_vector<float>(_numPoints,0.0f);
//        d_Vx = thrust::device_vector<float>(_numPoints,0.0f);
//        d_Vy = thrust::device_vector<float>(_numPoints,0.0f);
//        d_hash = thrust::device_vector<uint>(_numPoints,0);
//        d_cellOcc = thrust::device_vector<uint>(_gridRes*_gridRes,0);
//        d_scatterAdd = thrust::device_vector<uint>(_gridRes*_gridRes,0);
//    }

    ~ParticlesData();

    thrust::device_vector<float> d_Px;
    thrust::device_vector<float> d_Py;
    thrust::device_vector<float> d_prevPx;
    thrust::device_vector<float> d_prevPy;
    thrust::device_vector<float> d_Vx;
    thrust::device_vector<float> d_Vy;
    thrust::device_vector<uint> d_hash;
    thrust::device_vector<uint> d_cellOcc;
    thrust::device_vector<uint> d_scatterAdd;

    thrust::device_vector<uint> d_indexes;
};

#endif
