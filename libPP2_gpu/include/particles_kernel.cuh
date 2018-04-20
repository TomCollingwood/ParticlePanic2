#ifndef _PARTICLES_KERNEL_CUH_
#define _PARTICLES_KERNEL_CUH_


//__device__ float clamp(const float& value, const float& low, const float& high)
//{
//  return value < low ? low : (value > high ? high : value);
//}

__global__ void pointHash2D(unsigned int *hash,
                            float *Px,
                            float *Py,
                            const unsigned int N,
                            const unsigned int res);

__global__ void countCellOccupancyD(unsigned int *cellOcc,
                                    unsigned int *hash,
                                    unsigned int nCells,
                                    unsigned int nPoints);

__global__ void viscosityD(unsigned int _N,
                           unsigned int _gridRes,
                           float _iRadius,
                           float _timestep,
                           float * _P_x,
                           float * _P_y,
                           float * _V_x,
                           float * _V_y,
                           unsigned int * _d_hash,
                           unsigned int * _d_cellOcc,
                           unsigned int * _d_scatterAdd);

__global__ void densityD(unsigned int _N,
                         unsigned int _gridRes,
                         float _iRadius,
                         float _timestep,
                         float * _P_x,
                         float * _P_y,
                         float * _V_x,
                         float * _V_y,
                         unsigned int * _d_hash,
                         unsigned int * _d_cellOcc,
                         unsigned int * _d_scatterAdd);

__global__ void integrateD(unsigned int _N,
                           float _timestep,
                           float * _P_x,
                           float * _P_y,
                           float * _prevP_x,
                           float * _prevP_y,
                           float * _V_x,
                           float * _V_y);

__global__ void updateVelocityD(unsigned int _N,
                                float _timestep,
                                float * _P_x,
                                float * _P_y,
                                float * _prevP_x,
                                float * _prevP_y,
                                float * _V_x,
                                float * _V_y);

__global__ void addGravityD(unsigned int _N,
                            float * _V_x,
                            float * _V_y);

__global__ void boundaries(unsigned int _N,
                           float * _P_x,
                           float * _P_y);


#endif
