#include "PP2_gpu.h"

//-------------------------- KERNELS ----------------------------

__global__ void pointHash2D(unsigned int *hash,
                          const float *Px,
                          const float *Py,
                          //const float *Pz,
                          const unsigned int N,
                          const unsigned int res);

__global__ void countCellOccupancy(unsigned int *cellOcc,
                                   unsigned int *hash,
                                   unsigned int nCells,
                                   unsigned int nPoints);

__global__ void viscosity(unsigned int _N,
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

__global__ void integrate(unsigned int _N,
                          float _timestep,
                          float * _P_x,
                          float * _P_y,
                          float * _prevP_x,
                          float * _prevP_y,
                          float * _V_x,
                          float * _V_y);

__global__ void density(unsigned int _N,
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

__global__ void setNewVelocity(unsigned int _N,
                               float _timestep,
                               float * _P_x,
                               float * _P_y,
                               float * _prevP_x,
                               float * _prevP_y,
                               float * _V_x,
                               float * _V_y);


