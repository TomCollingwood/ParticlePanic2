
#include <iostream>
#include <cstdlib>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

// Needed for output functions within the kernel
#include <stdio.h>

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include<sys/time.h>

#include <sstream> 
#include <fstream>

// My own include function to generate some randomness
#include "random.cuh"


/// The number of points to generate within 0,1
// #define m_num_points 40000

// /// The resolution of our grid (dependent on the radius of influence of each point)
// #define m_grid_resolution 4

/// The null hash indicates the point isn't in the grid (this shouldn't happen if your extents are correctly chosen)
#define NULL_HASH UINT_MAX

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//------------------------------------TOMS KERNELS-----------------------------------------------

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
                          float *Px,
                          float *Py,
                          const unsigned int N,
                          const unsigned int res) {
    // Compute the index of this thread: i.e. the point we are testing
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Note that finding the grid coordinates are much simpler if the grid is over the range [0,1] in
        // each dimension and the points are also in the same space.

        float low = 0.0f;
        float high = 0.999f;
        Px[idx] = Px[idx] < low ? low : (Px[idx] > high ? high : Px[idx]);
        Py[idx] = Py[idx] < low ? low : (Py[idx] > high ? high : Py[idx]);

        int gridPos[2];
        gridPos[0] = floor(Px[idx] * float(res));
        gridPos[1] = floor(Py[idx] * float(res));
        //gridPos[2] = floor(Pz[idx] * res);

        // Write out the hash value if the point is within range [0,1], else write NULL_HASH
        hash[idx] = gridPos[0] * res + gridPos[1];

        // Uncomment the lines below for debugging. Not recommended for 4mil points!
        // printf("pointHash<<<%d>>>: P=[%f,%f] gridPos=[%d,%d] hash=%d\n",
        //     idx, Px[idx], Py[idx],
        //     gridPos[0], gridPos[1], hash[idx]);
    }
}

/**
  * Compute the grid cell occupancy from the input vector of grid hash values. Note that the hash[]
  * vector doesn't need to be presorted, but performance will probably improve if the memory is
  * contiguous.
  * \param cellOcc A vector, size GRID_RES^3, which will contain the occupancy of each cell
  * \param hash A vector, size m_num_points, which contains the hash of the grid cell of this point
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

    // int nCells = _gridRes*_gridRes;

    unsigned int hash =  _d_hash[idx];
    unsigned int startIndex = _d_scatterAdd[hash];
    unsigned int endIndex = startIndex + _d_cellOcc[hash];
    if(endIndex>=_N) endIndex = _N-1;

    for(int otherid = 0; otherid<_N;++otherid)
    {
        if(otherid>=_N) break;
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
                atomicAdd(&(_V_x[idx]),-impulseX);
                atomicAdd(&(_V_y[idx]),-impulseY);
                atomicAdd(&(_V_x[otherid]),impulseX);
                atomicAdd(&(_V_y[otherid]),impulseY);
            }
        }
    }
}

__global__ void boundaries(unsigned int _N,
                           float * _P_x,
                           float * _P_y)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > _N) return;

    float low = 0.0f;
    float high = 1.0f;
    _P_x[idx] = _P_x[idx] < low ? low : (_P_x[idx] > high ? high : _P_x[idx]);
    _P_y[idx] = _P_y[idx] < low ? low : (_P_y[idx] > high ? high : _P_y[idx]);
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

    // int nCells = _gridRes*_gridRes;

    unsigned int hash =  _d_hash[idx];
    unsigned int startIndex = _d_scatterAdd[hash];
    unsigned int endIndex = startIndex+_d_cellOcc[hash];
    if(endIndex>=_N) endIndex=_N-1;

    float density = 0.0f;
    float nearDensity = 0.0f;

    for(int otherid = 0; otherid<_N;++otherid)
    {
        if(otherid>=_N) break;
        if(otherid<0) break;
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

    float p0 = 5.0f;
    float k = 0.004f;
    float knear = 0.01f;

    float P = k*(density - p0);
    float Pnear = knear * nearDensity;

    float dx = 0.0f;
    float dy = 0.0f;

    for(int otherid = 0; otherid<_N;++otherid)
    {
        if(otherid>=_N) break;
        if(otherid<0) break;
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

    _P_x[idx] = _P_x[idx] + _V_x[idx] * _timestep;
    _P_y[idx] = _P_y[idx] +_V_y[idx] * _timestep;

    float low = 0.0f;
    float high = 0.999f;
    _P_x[idx] = _P_x[idx] < low ? low : (_P_x[idx] > high ? high : _P_x[idx]);
    _P_y[idx] = _P_y[idx] < low ? low : (_P_y[idx] > high ? high : _P_y[idx]);
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

    // printf("updateVelocity<<<%d>>>: P=[%f,%f] V=[%f,%f]\n",
    //     idx, _P_x[idx], _P_y[idx],
    //     _V_x[idx], _V_y[idx]);
}

__global__ void addGravityD(unsigned int _N,
                           float * _V_x,
                           float * _V_y)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > _N) return;
    _V_y[idx]+=-0.008f;
}

void dumpToGeo(const thrust::device_vector<float> &Px,
               const thrust::device_vector<float> &Py,
               const uint cnt) 
{
    char fname[150];

    std::sprintf(fname,"geo/SPH_GPU.%03d.geo",cnt);
    // we will use a stringstream as it may be more efficient
    std::stringstream ss;
    std::ofstream file;
    file.open(fname);
    if (!file.is_open())
    {
        std::cerr << "failed to Open file "<<fname<<'\n';
        exit(EXIT_FAILURE);
    }
    // write header see here http://www.sidefx.com/docs/houdini15.0/io/formats/geo
    ss << "PGEOMETRY V5\n";
    ss << "NPoints " << Px.size() << " NPrims 1\n";
    ss << "NPointGroups 0 NPrimGroups 1\n";
    // this is hard coded but could be flexible we have 1 attrib which is Colour
    ss << "NPointAttrib 1  NVertexAttrib 0 NPrimAttrib 2 NAttrib 0\n";
    // now write out our point attrib this case Cd for diffuse colour
    ss <<"PointAttrib \n";
    // default the colour to white
    ss <<"Cd 3 float 1 1 1\n";
    // now we write out the particle data in the format
    // x y z 1 (attrib so in this case colour)
    for(unsigned int i=0; i<Px.size(); ++i)
    {
        ss<<Px[i]<<" "<<Py[i]<<" "<<0 << " 1 ";
        ss<<"("<<1<<" "<<1<<" "<<1<<")\n";
    }

    // now write out the index values
    ss<<"PrimitiveAttrib\n";
    ss<<"generator 1 index 1 location1\n";
    ss<<"dopobject 1 index 1 /obj/AutoDopNetwork:1\n";
    ss<<"Part "<<Px.size()<<" ";
    for(size_t i=0; i<Px.size(); ++i)
    {
        ss<<i<<" ";
    }
    ss<<" [0	0]\n";
    ss<<"box_object1 unordered\n";
    ss<<"1 1\n";
    ss<<"beginExtra\n";
    ss<<"endExtra\n";
    // dump string stream to disk;
    file<<ss.rdbuf();
    file.close();
}

/**
 * Host main routine
 */
int main(int argc, char **argv) {

    int m_num_points = 10000;
    int m_num_frames = 60;
    float m_interactionRadius = 0.005f;
    float m_timestep = 0.002f;
    int m_grid_resolution = 4;
    if(argc>1) m_num_points=atoi(argv[1]);
    if(argc>2) m_num_frames=atoi(argv[2]);
    if(argc>3) m_interactionRadius=atof(argv[3]);
    if(argc>4) m_timestep=atof(argv[4]);
    if(argc>5) m_grid_resolution=atoi(argv[5]);

    thrust::device_vector<float> d_Px(m_num_points);
    thrust::device_vector<float> d_Py(m_num_points);
    thrust::device_vector<float> d_prevPx(m_num_points);
    thrust::device_vector<float> d_prevPy(m_num_points);
    thrust::device_vector<float> d_Vx(m_num_points,0.0f);
    thrust::device_vector<float> d_Vy(m_num_points,0.0f);

    //------------------DAMBREAKER 100----------------------
    if(m_num_points==100)
    {
        srand(42);
        for(int x = 0; x<5; ++x)
        {
            for(int y =0; y<20; ++y)
            {
                float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float yr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                xr = 0.5f-xr;
                yr = 0.5f-yr;
                d_Px[x+y*5] = float(x)*(1.0f/20.0f)+xr*0.01f;
                d_Py[x+y*5] = float(y)*(1.0f/20.0f)+yr*0.01f;
                d_prevPx[x+y*5] = float(x)*(1.0f/20.0f)+xr*0.01f;
                d_prevPy[x+y*5] = float(y)*(1.0f/20.0f)+yr*0.01f;
            }
        }
    }

    //------------------DAMBREAKER 10,000----------------------
    else if (m_num_points==10000)
    {
        srand(42);
        for(int x = 0; x<50; ++x)
        {
            for(int y =0; y<200; ++y)
            {
                float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float yr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                xr = 0.5f-xr;
                yr = 0.5f-yr;
                d_Px[x+y*50] = float(x)*(1.0f/200.0f)+xr*0.001f;
                d_Py[x+y*50] = float(y)*(1.0f/200.0f)+yr*0.001f;
                d_prevPx[x+y*50] = float(x)*(1.0f/200.0f)+xr*0.001f;
                d_prevPy[x+y*50] = float(y)*(1.0f/200.0f)+yr*0.001f;
            }
        }
    }

    //------------------DAMBREAKER 1,000,000----------------------
    else if (m_num_points==1000000)
    {
        srand(42);
        for(int x = 0; x<500; ++x)
        {
            for(int y =0; y<2000; ++y)
            {
                float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float yr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                xr = 0.5f-xr;
                yr = 0.5f-yr;
                d_Px[x+y*500] = float(x)*(1.0f/2000.0f)+xr*0.0001f;
                d_Py[x+y*500] = float(y)*(1.0f/2000.0f)+yr*0.0001f;
                d_prevPx[x+y*500] = float(x)*(1.0f/2000.0f)+xr*0.0001f;
                d_prevPy[x+y*500] = float(y)*(1.0f/2000.0f)+yr*0.0001f;
            }
        }
    }

    // This vector will hold the grid cell occupancy (set to zero)
    thrust::device_vector<unsigned int> d_cellOcc(m_grid_resolution*m_grid_resolution, 0);

    thrust::device_vector<unsigned int> d_scatterAdd(m_grid_resolution*m_grid_resolution, 0);

    // This vector will hold our hash values, one for each point
    thrust::device_vector<unsigned int> d_hash(m_num_points);
    //thrust::copy(d_hash.begin(), d_hash.end(), std::ostream_iterator<unsigned int>(std::cout, " "));

    // Typecast some raw pointers to the data so we can access them with CUDA functions
    unsigned int * d_hash_ptr = thrust::raw_pointer_cast(&d_hash[0]);
    unsigned int * d_cellOcc_ptr = thrust::raw_pointer_cast(&d_cellOcc[0]);
    unsigned int * d_scatterAdd_ptr = thrust::raw_pointer_cast(&d_scatterAdd[0]);
    float * d_Px_ptr = thrust::raw_pointer_cast(&d_Px[0]);
    float * d_Py_ptr = thrust::raw_pointer_cast(&d_Py[0]);
    float * d_prevPx_ptr = thrust::raw_pointer_cast(&d_prevPx[0]);
    float * d_prevPy_ptr = thrust::raw_pointer_cast(&d_prevPy[0]);
    float * d_Vx_ptr = thrust::raw_pointer_cast(&d_Vx[0]);
    float * d_Vy_ptr = thrust::raw_pointer_cast(&d_Vy[0]);

    int frame = 0;

    //Begin hash

    //---------------------------HASHING---------------------------------------
    unsigned int nThreads = 1024;
    unsigned int nBlocks = m_num_points / nThreads + 1;
    cudaThreadSynchronize();
    pointHash2D<<<nBlocks, nThreads>>>(d_hash_ptr, d_Px_ptr, d_Py_ptr,
                                        m_num_points,
                                        m_grid_resolution);
    cudaThreadSynchronize();

    // Now we can sort our points to ensure that points in the same grid cells occupy contiguous memory
    thrust::sort_by_key(d_hash.begin(), d_hash.end(),
                        thrust::make_zip_iterator(
                            thrust::make_tuple( d_Px.begin(), d_Py.begin(), d_prevPx.begin(),d_prevPy.begin(),d_Vx.begin(),d_Vy.begin())));
    cudaThreadSynchronize();

    // Now we can count the number of points in each grid cell
    countCellOccupancyD<<<nBlocks, nThreads>>>(d_cellOcc_ptr, d_hash_ptr, d_cellOcc.size(), d_hash.size());

    // DONE CORRECTLY
    cudaThreadSynchronize();
    thrust::exclusive_scan(d_cellOcc.begin(),d_cellOcc.end(),d_scatterAdd.begin());
    cudaThreadSynchronize();


    for(int i = 0; i<m_num_frames ; ++i)
    {


        //-------------------------TIMING----------------------------------------
        struct timeval tim;
        double t1, t2;
        gettimeofday(&tim, NULL);
        t1=tim.tv_sec+(tim.tv_usec/1000000.0);

        //-------------------------GRAVITY----------------------------------------
        addGravityD<<<nBlocks, nThreads>>>(m_num_points,d_Vx_ptr,d_Vy_ptr);
        gpuErrchk( cudaPeekAtLastError() );
        cudaThreadSynchronize();

        //-------------------------VISCOSITY--------------------------------------
        viscosityD<<<nBlocks, nThreads>>>(m_num_points,
                                            m_grid_resolution,
                                            m_interactionRadius,
                                            m_timestep,
                                            d_Px_ptr,
                                            d_Py_ptr,
                                            d_Vx_ptr,
                                            d_Vy_ptr,
                                            d_hash_ptr,
                                            d_cellOcc_ptr,
                                            d_scatterAdd_ptr);
        gpuErrchk( cudaPeekAtLastError() );
        cudaThreadSynchronize();  

        //--------------------------INTEGRATE-------------------------------------
        integrateD<<<nBlocks,nThreads>>>(m_num_points,
                                        m_timestep,
                                        d_Px_ptr,
                                        d_Py_ptr,
                                        d_prevPx_ptr,
                                        d_prevPy_ptr,
                                        d_Vx_ptr,
                                        d_Vy_ptr);
        gpuErrchk( cudaPeekAtLastError() );  
        cudaThreadSynchronize();
        boundaries<<<nBlocks,nThreads>>>(m_num_points,
                                            d_Px_ptr,
                                            d_Py_ptr);
        cudaThreadSynchronize();

        //---------------------------HASHING---------------------------------------
        pointHash2D<<<nBlocks, nThreads>>>(d_hash_ptr, d_Px_ptr, d_Py_ptr,
                                            m_num_points,
                                            m_grid_resolution);
        cudaThreadSynchronize();

        // Now we can sort our points to ensure that points in the same grid cells occupy contiguous memory
        thrust::sort_by_key(d_hash.begin(), d_hash.end(),
                            thrust::make_zip_iterator(
                                thrust::make_tuple( d_Px.begin(), d_Py.begin(), d_prevPx.begin(),d_prevPy.begin(),d_Vx.begin(),d_Vy.begin())));
        cudaThreadSynchronize();

        d_cellOcc.assign(m_grid_resolution*m_grid_resolution,0);
        // Now we can count the number of points in each grid cell
        countCellOccupancyD<<<nBlocks, nThreads>>>(d_cellOcc_ptr, d_hash_ptr, d_cellOcc.size(), d_hash.size());

        // DONE CORRECTLY
        cudaThreadSynchronize();
        d_scatterAdd.resize(m_grid_resolution*m_grid_resolution,0);
        thrust::exclusive_scan(d_cellOcc.begin(),d_cellOcc.end(),d_scatterAdd.begin());
        cudaThreadSynchronize();


        // Only dump the debugging information if we have a manageable number of points.
        // if (m_num_points <= 100) {
        //     std::cout << "\n";
        //     thrust::copy(d_hash.begin(), d_hash.end(), std::ostream_iterator<unsigned int>(std::cout, " "));
        //     std::cout << "\n";
        //     thrust::copy(d_cellOcc.begin(), d_cellOcc.end(), std::ostream_iterator<unsigned int>(std::cout, " "));
        //     std::cout << "\n";
        //     thrust::copy(d_scatterAdd.begin(), d_scatterAdd.end(), std::ostream_iterator<unsigned int>(std::cout, " "));
        //     std::cout << "\n";
        // }


        //---------------------------DOUBLE DENSITY------------------------------

        gpuErrchk( cudaPeekAtLastError() );
        densityD<<<nBlocks,nThreads>>>(m_num_points,
                                        m_grid_resolution,
                                        m_interactionRadius,
                                        m_timestep,
                                        d_Px_ptr,
                                        d_Py_ptr,
                                        d_Vx_ptr,
                                        d_Vy_ptr,
                                        d_hash_ptr,
                                        d_cellOcc_ptr,
                                        d_scatterAdd_ptr);

        gpuErrchk( cudaPeekAtLastError() );                               
        cudaThreadSynchronize();
        boundaries<<<nBlocks,nThreads>>>(m_num_points,
                                            d_Px_ptr,
                                            d_Py_ptr);
        cudaThreadSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        //---------------------------UPDATE VELOCITY-------------------------------
        updateVelocityD<<<nBlocks,nThreads>>>(m_num_points,
                                                m_timestep,
                                                d_Px_ptr,
                                                d_Py_ptr,
                                                d_prevPx_ptr,
                                                d_prevPy_ptr,
                                                d_Vx_ptr,
                                                d_Vy_ptr);
        gpuErrchk( cudaPeekAtLastError() );
        cudaThreadSynchronize();

        gettimeofday(&tim, NULL);
        t2=tim.tv_sec+(tim.tv_usec/1000000.0);
        std::cout << "Frame "<<frame <<" done. Grid sorted and simulated "<<m_num_points<<" points into grid of "<<m_grid_resolution*m_grid_resolution<<" cells in " << t2-t1 << "s\n";

        //------------------ EXPORT TO GEO ------------------------------
        cudaDeviceSynchronize();
        // gettimeofday(&tim, NULL);
        // t1=tim.tv_sec+(tim.tv_usec/1000000.0);
        // thrust::host_vector<float> h_Px(m_num_points);
        // thrust::host_vector<float> h_Py(m_num_points);

        // thrust::copy(d_Px.begin(), d_Px.end(), h_Px.begin());
        // thrust::copy(d_Py.begin(), d_Py.end(), h_Py.begin());

        // gettimeofday(&tim, NULL);
        // t2=tim.tv_sec+(tim.tv_usec/1000000.0);
        // std::cout << "Copied "<<m_num_points<<" points from device to host in " << t2-t1 << "s\n";
        //dumpToObj(d_Px,d_Py,frame);
        dumpToGeo(d_Px,d_Py,frame);

        frame++;
    }

    return 0;
}

