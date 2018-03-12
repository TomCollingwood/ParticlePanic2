#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <time.h>
#include <iostream>


//// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "PP2_gpu.h"

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
__global__ void PP2_GPU::pointHash2D(unsigned int *hash,
                          const float *Px,
                          const float *Py,
                          //const float *Pz,
                          const unsigned int N,
                          const unsigned int res) {
    // Compute the index of this thread: i.e. the point we are testing
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Note that finding the grid coordinates are much simpler if the grid is over the range [0,1] in
        // each dimension and the points are also in the same space.
        int gridPos[3];
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
__global__ void PP2_GPU::countCellOccupancy(unsigned int *cellOcc,
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

void PP2_GPU::initData()
{
    d_Px = thrust::device_vector<float>();
    d_Py = thrust::device_vector<float>();
    d_Px_ptr = thrust::raw_pointer_cast(&d_Px[0]);
    d_Py_ptr = thrust::raw_pointer_cast(&d_Py[0]);

    d_Vx = thrust::device_vector<float>();
    d_Vy = thrust::device_vector<float>();
    d_Vx_ptr = thrust::raw_pointer_cast(&d_Vx[0]);
    d_Vy_ptr = thrust::raw_pointer_cast(&d_Vy[0]);

    d_hash_ptr = thrust::raw_pointer_cast(&d_hash[0]);
    d_cellOcc_ptr = thrust::raw_pointer_cast(&d_cellOcc[0]);
}

void PP2_GPU::hashOccSort()
{
    unsigned int nThreads = 1024;
    unsigned int nBlocks = m_numPoints / nThreads + 1;

//    pointHash2D<<<nBlocks, nThreads>>>(d_hash_ptr, d_Px_ptr, d_Py_ptr,
//                                         m_numPoints,
//                                         m_gridResolution);

//    thrust::sort_by_key(d_hash.begin(), d_hash.end(),
//                            thrust::make_zip_iterator(
//                                thrust::make_tuple( d_Px.begin(), d_Py.begin(), d_Vx.begin(),d_Vy.begin())));
}
