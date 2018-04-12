#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

//// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>

/**
  * Find the cell hash of each point. The hash is returned as the mapping of a point index to a cell.
  * If the point isn't inside any cell, it is set to NULL_HASH. This may have repercussions later in
  * the code.
  * \param Px The array of x values
  * \param Py The array of y values
  * \param Pz the array of z values
  * \param hash The array of hash output
  * \param N The number of points (dimensions of Px,Py,Pz and hash)
  * \param res The resolution of our grid.
  */
__global__ void pointHash2D(unsigned int *hash,
                          const float *Px,
                          const float *Py,
                          const unsigned int N,
                          const unsigned int res) {
    // Compute the index of this thread: i.e. the point we are testing
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Note that finding the grid coordinates are much simpler if the grid is over the range [0,1] in
        // each dimension and the points are also in the same space.
        int gridPos[2];
        gridPos[0] = floor(Px[idx] * res);
        gridPos[1] = floor(Py[idx] * res);
        //gridPos[2] = floor(Pz[idx] * res);

        // Test to see if all of the points are inside the grid
        bool isInside = true;
        unsigned int i;
        for (i=0; i<3; ++i)
            if ((gridPos[i] < 0) || (gridPos[i] > res)) {
                isInside = false;
            }

        // Write out the hash value if the point is within range [0,1], else write NULL_HASH
        if (isInside) {
            hash[idx] = gridPos[0] * res + gridPos[1];
        } else {
            hash[idx] = UINT_MAX;
        }
        // Uncomment the lines below for debugging. Not recommended for 4mil points!
        //printf("pointHash<<<%d>>>: P=[%f,%f,%f] gridPos=[%d,%d,%d] hash=%d\n",
        //       idx, Px[idx], Py[idx], Pz[idx],
        //       gridPos[0], gridPos[1], gridPos[2], hash[idx]);
    }
}

/**
  * Compute the grid cell occupancy from the input vector of grid hash values. Note that the hash[]
  * vector doesn't need to be presorted, but performance will probably improve if the memory is
  * contiguous.
  * \param cellOcc A vector, size GRID_RES^3, which will contain the occupancy of each cell
  * \param hash A vector, size NUM_POINTS, which contains the hash of the grid cell of this point
  * \param nCells The size of the cellOcc vector (GRID_RES^3)
  * \param nPoints The number of points (size of hash)
  */
__global__ void countCellOccupancyD(unsigned int *cellOcc,
                                   unsigned int *hash,
                                   unsigned int nCells,
                                   unsigned int nPoints) {
    // Compute the index of this thread: i.e. the point we are testing
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform a sanity check and then atomic add to the occupancy count of the relevant cell
    if ((idx < nPoints) && (hash[idx] < nCells)) {
        atomicAdd(&(cellOcc[hash[idx]]), 1);
    }
}

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
                          unsigned int * _d_scatterAdd)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > _N) return;

    unsigned int hashCentre =  _d_hash[idx];

    unsigned int imin, imax, jmin, jmax;
    if(hashCentre%_gridRes ==0) imin=0;
    if(hashCentre% _gridRes == _gridRes-1) imax=0;

    if(hashCentre/ _gridRes==0) jmax = 0;
    if(hashCentre/ _gridRes== _gridRes-1) jmin=0;

    for(int i = imin; i<=imax; ++i)
    {
        for(int j = jmin; j<=jmax; ++j)
        {
            if(i==0 && j==0) break;
            unsigned int otherHash = (hashCentre+i)+j*_gridRes;
            unsigned int startIndex = _d_scatterAdd[otherHash+1];
            unsigned int howMany = _d_cellOcc[otherHash];

            for(int otherid = startIndex; otherid<startIndex+howMany;++otherid)
            {
                float diffX = _P_x[otherid]-_P_x[idx];
                float diffY = _P_y[otherid]-_P_y[idx];
                float mag = sqrtf(diffX*diffX + diffY*diffY);
                float q = mag / _iRadius;
                if(q<1 && q!=0)
                {
                    float diffXnorm = diffX/mag;
                    float diffYnorm = diffY/mag;
                    float diffXV = _V_x[idx] - _V_x[otherid];
                    float diffYV = _V_y[idx] - _V_y[otherid];
                    float u = (diffXV*diffXnorm) + (diffYV*diffYnorm);
                    if(u>0)
                    {
                        float sig = 0.05f;
                        float bet = 0.1f;
                        float h = (1-q)*(sig*u + bet*u*u)*_timestep;
                        float impulseX = diffXnorm*h;
                        float impulseY = diffYnorm*h;
                        _V_x[idx]= _V_x[idx] + impulseX;
                        _V_y[idx]= _V_y[idx] + impulseY;
                        atomicAdd(&(_V_x[idx]),-impulseX);
                        atomicAdd(&(_V_y[idx]),-impulseY);
                        atomicAdd(&(_V_x[otherid]),impulseX);
                        atomicAdd(&(_V_y[otherid]),impulseY);
                    }
                }
            }
        }
    }
}

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
                        unsigned int * _d_scatterAdd)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > _N) return;

    unsigned int hashCentre =  _d_hash[idx];

    unsigned int imin, imax, jmin, jmax;
    if(hashCentre%_gridRes ==0) imin=0;
    if(hashCentre% _gridRes == _gridRes-1) imax=0;

    if(hashCentre/ _gridRes==0) jmax = 0;
    if(hashCentre/ _gridRes== _gridRes-1) jmin=0;

    float density = 0.0f;
    float nearDensity = 0.0f;

    for(int i = imin; i<=imax; ++i)
    {
        for(int j = jmin; j<=jmax; ++j)
        {
            if(i==0 && j==0) break;
            unsigned int otherHash = (hashCentre+i)+j*_gridRes;
            unsigned int startIndex = _d_scatterAdd[otherHash+1];
            unsigned int howMany = _d_cellOcc[otherHash];

            for(int otherid = startIndex; otherid<startIndex+howMany;++otherid)
            {
                float Rx = _P_x[otherid] - _P_x[idx];
                float Ry = _P_y[otherid] - _P_y[idx];
                float magR = sqrtf(Rx*Rx + Ry*Ry);
                float q = magR / _iRadius;
                if(q<1 && q!=0)
                {
                    density+=(1.0f-q)*(1.0f-q);
                    nearDensity+=(1.0f-q)*(1.0f-q)*(1.0f-q);
                }
            }
        }
    }
    float p0 = 5.0f;
    float k = 0.004f;
    float knear = 0.01f;

    float P = k*(density - p0);
    float Pnear = knear * nearDensity;

    float dx = 0.0f;
    float dy = 0.0f;
    for(int i = imin; i<=imax; ++i)
    {
        for(int j = jmin; j<=jmax; ++j)
        {
            if(i==0 && j==0) break;
            unsigned int otherHash = (hashCentre+i)+j*_gridRes;
            unsigned int startIndex = _d_scatterAdd[otherHash+1];
            unsigned int howMany = _d_cellOcc[otherHash];

            for(int otherid = startIndex; otherid<startIndex+howMany;++otherid)
            {
                float Rx = _P_x[otherid] - _P_x[idx];
                float Ry = _P_y[otherid] - _P_y[idx];
                float magR = sqrtf(Rx*Rx + Ry*Ry);
                float q = magR / _iRadius;
                if(q<1 && q!=0)
                {
                    float Rxnorm = Rx / magR;
                    float Rynorm = Ry / magR;
                    float he = (_timestep*_timestep*(P*(1.0f-q))+Pnear*(1.0f-q)*(1.0f-q));
                    float Dx = Rxnorm*he;
                    float Dy = Rynorm*he;
                    atomicAdd(&(_P_x[otherid]), Dx/2.0f);
                    atomicAdd(&(_P_y[otherid]), Dy/2.0f);
                    dx-=Dx/2.0f;
                    dy-=Dy/2.0f;
                }
            }
        }
    }
    atomicAdd(&(_P_x[idx]), dx);
    atomicAdd(&(_P_y[idx]), dy);
}

__global__ void integrateD(unsigned int _N,
                          float _timestep,
                          float * _P_x,
                          float * _P_y,
                          float * _prevP_x,
                          float * _prevP_y,
                          float * _V_x,
                          float * _V_y)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > _N) return;

    _prevP_x[idx] = _P_x[idx];
    _prevP_y[idx] = _P_y[idx];

    _P_x[idx] = _V_x[idx] * _timestep;
    _P_y[idx] = _V_y[idx] * _timestep;

    // boundaries
    if(_P_x[idx]<0.0f) _P_x[idx]=0.0f;
    if(_P_x[idx]>1.0f) _P_x[idx]=1.0f;
    if(_P_y[idx]<0.0f) _P_y[idx]=0.0f;
    if(_P_y[idx]>1.0f) _P_y[idx]=1.0f;

}

__global__ void updateVelocityD(unsigned int _N,
                               float _timestep,
                               float * _P_x,
                               float * _P_y,
                               float * _prevP_x,
                               float * _prevP_y,
                               float * _V_x,
                               float * _V_y)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > _N) return;

    _V_x[idx] = (_P_x[idx] - _prevP_x[idx])/_timestep;
    _V_y[idx] = (_P_y[idx] - _prevP_y[idx])/_timestep;
}

__global__ void addGravityD(unsigned int _N,
                           float * _V_x,
                           float * _V_y)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > _N) return;
    _V_y[idx]+=-0.0098f;
}


#endif
