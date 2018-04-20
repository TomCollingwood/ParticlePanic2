#include <stdio.h>
#include <time.h>
#include <iostream>
#include <math.h>
#include <cstdlib>

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

#include "World_gpu.cuh"
#include "particles_kernel.cuh"
#include "particles_data.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//extern "C"
//{

ParticlesData * initializeParticlesData(int _numPoints, int _gridRes)
{
    ParticlesData * _data = new ParticlesData();

    int nCells = _gridRes*_gridRes;

    gpuErrchk( cudaPeekAtLastError() );

    thrust::host_vector<float> h_Px(_numPoints);
    thrust::host_vector<float> h_Py(_numPoints);
    thrust::host_vector<float> h_prevPx(_numPoints);
    thrust::host_vector<float> h_prevPy(_numPoints);

    if(_numPoints<=100)
    {
        //------------------DAMBREAKER 100----------------------
        srand(42);
        for(int x = 0; x<5; ++x)
        {
            for(int y =0; y<20; ++y)
            {
                if(x+y*5 >= _numPoints) break;
                float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float yr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                xr = 0.5f-xr;
                yr = 0.5f-yr;
                h_Px[x+y*5] = float(x)*(1.0f/20.0f)+xr*0.01f;
                h_Py[x+y*5] = float(y)*(1.0f/20.0f)+yr*0.01f;
                h_prevPx[x+y*5] = float(x)*(1.0f/20.0f)+xr*0.01f;
                h_prevPy[x+y*5] = float(y)*(1.0f/20.0f)+yr*0.01f;
            }
        }
    }
    else if (_numPoints<=10000)
    {
        //------------------DAMBREAKER 10,000----------------------
        srand(42);
        for(int x = 0; x<50; ++x)
        {
            for(int y =0; y<200; ++y)
            {
                if(x+y*50 >= _numPoints) break;
                float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float yr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                xr = 0.5f-xr;
                yr = 0.5f-yr;
                h_Px[x+y*50] = float(x)*(1.0f/200.0f)+xr*0.001f;
                h_Py[x+y*50] = float(y)*(1.0f/200.0f)+yr*0.001f;
                h_prevPx[x+y*50] = float(x)*(1.0f/200.0f)+xr*0.001f;
                h_prevPy[x+y*50] = float(y)*(1.0f/200.0f)+yr*0.001f;
            }
        }
    }
    else if (_numPoints<=1000000)
    {
        //------------------DAMBREAKER 1,000,000----------------------
        srand(42);
        for(int x = 0; x<500; ++x)
        {
            for(int y =0; y<2000; ++y)
            {
                if(x+y*500 >= _numPoints) break;
                float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float yr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                xr = 0.5f-xr;
                yr = 0.5f-yr;
                h_Px[x+y*500] = float(x)*(1.0f/2000.0f)+xr*0.0001f;
                h_Py[x+y*500] = float(y)*(1.0f/2000.0f)+yr*0.0001f;
                h_prevPx[x+y*500] = float(x)*(1.0f/2000.0f)+xr*0.0001f;
                h_prevPy[x+y*500] = float(y)*(1.0f/2000.0f)+yr*0.0001f;
            }
        }
    }

    thrust::copy(h_Px.begin(),h_Px.end(),_data->d_Px.begin());
    thrust::copy(h_Py.begin(),h_Py.end(),_data->d_Py.begin());
    thrust::copy(h_prevPx.begin(),h_prevPx.end(),_data->d_prevPx.begin());
    thrust::copy(h_prevPy.begin(),h_prevPy.end(),_data->d_prevPy.begin());

    _data->d_Vx = thrust::device_vector<float>(_numPoints,0.0f);
    _data->d_Vy = thrust::device_vector<float>(_numPoints,0.0f);

    _data->d_hash = thrust::device_vector<unsigned int>(_numPoints,0);
    _data->d_cellOcc = thrust::device_vector<unsigned int>(nCells,0);
    _data->d_scatterAdd = thrust::device_vector<unsigned int>(nCells,0);

    return _data;
}

void dumpToGeo(ParticlesData *_data,
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
    ss << "NPoints " << _data->d_Px.size() << " NPrims 1\n";
    ss << "NPointGroups 0 NPrimGroups 1\n";
    // this is hard coded but could be flexible we have 1 attrib which is Colour
    ss << "NPointAttrib 1  NVertexAttrib 0 NPrimAttrib 2 NAttrib 0\n";
    // now write out our point attrib this case Cd for diffuse colour
    ss <<"PointAttrib \n";
    // default the colour to white
    ss <<"Cd 3 float 1 1 1\n";
    // now we write out the particle data in the format
    // x y z 1 (attrib so in this case colour)
    for(unsigned int i=0; i<_data->d_Px.size(); ++i)
    {
        ss<<_data->d_Px[i]<<" "<<_data->d_Py[i]<<" "<<0 << " 1 ";
        ss<<"("<<1<<" "<<1<<" "<<1<<")\n";
    }

    // now write out the index values
    ss<<"PrimitiveAttrib\n";
    ss<<"generator 1 index 1 location1\n";
    ss<<"dopobject 1 index 1 /obj/AutoDopNetwork:1\n";
    ss<<"Part "<<_data->d_Px.size()<<" ";
    for(size_t i=0; i<_data->d_Px.size(); ++i)
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


void pointHash(int _num_points,
             int _gridRes,
             ParticlesData * _data
             )
{
    if(_num_points==0) return;
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _num_points / nThreads + 1;
    unsigned int nCells = _gridRes*_gridRes;

    gpuErrchk( cudaPeekAtLastError() );

    pointHash2D<<<nBlocks, nThreads>>>(thrust::raw_pointer_cast(_data->d_hash.data()),
                                       thrust::raw_pointer_cast(_data->d_Px.data()),
                                       thrust::raw_pointer_cast(_data->d_Py.data()),
                                       _num_points,
                                       _gridRes);

    gpuErrchk( cudaPeekAtLastError() );
}

void sortHash(ParticlesData * _data)
{
    gpuErrchk( cudaPeekAtLastError() );
    auto tuple = thrust::make_tuple( _data->d_Px.begin(), _data->d_Py.begin(), _data->d_Vx.begin(), _data->d_Vy.begin(), _data->d_prevPx.begin(), _data->d_prevPy.begin());
    gpuErrchk( cudaPeekAtLastError() );
    auto zippy = thrust::make_zip_iterator(tuple);
    gpuErrchk( cudaPeekAtLastError() );
    thrust::sort_by_key(_data->d_hash.begin(), _data->d_hash.end(), zippy); // bad alloc here
    gpuErrchk( cudaPeekAtLastError() );
}

void countCellOcc(int _num_points,
               int _gridRes,
               ParticlesData * _data)
{
    if(_num_points==0) return;
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _num_points / nThreads + 1;
    unsigned int nCells = _gridRes*_gridRes;
    countCellOccupancyD<<<nBlocks, nThreads>>>((uint *)thrust::raw_pointer_cast(_data->d_cellOcc.data()),
                                               (uint *)thrust::raw_pointer_cast(_data->d_hash.data()),
                                               nCells,
                                               _num_points);
}

void exclusiveScan(ParticlesData * _data)
{
    gpuErrchk( cudaPeekAtLastError() );

    thrust::exclusive_scan(_data->d_cellOcc.begin(),_data->d_cellOcc.end(),_data->d_scatterAdd.begin());
    gpuErrchk( cudaPeekAtLastError() );

}

void hashOccSort(int _num_points,
             int _gridRes,
             ParticlesData * _data
             )
{
    if(_num_points==0) return;
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _num_points / nThreads + 1;
    unsigned int nCells = _gridRes*_gridRes;

    gpuErrchk( cudaPeekAtLastError() );

    pointHash2D<<<nBlocks, nThreads>>>(thrust::raw_pointer_cast(_data->d_hash.data()),
                                       thrust::raw_pointer_cast(_data->d_Px.data()),
                                       thrust::raw_pointer_cast(_data->d_Py.data()),
                                       _num_points,
                                       _gridRes);

    gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    //if(cudaSuccess==cudaThreadSynchronize()) std::cout<<"it works"<<std::endl;

    gpuErrchk( cudaPeekAtLastError() ); // stops here
    auto tuple = thrust::make_tuple( _data->d_Px.begin(), _data->d_Py.begin(), _data->d_Vx.begin(), _data->d_Vy.begin(), _data->d_prevPx.begin(), _data->d_prevPy.begin());
    gpuErrchk( cudaPeekAtLastError() );
    auto zippy = thrust::make_zip_iterator(tuple);
    gpuErrchk( cudaPeekAtLastError() );
    thrust::sort_by_key(_data->d_hash.begin(), _data->d_hash.end(), zippy); // bad alloc here

    gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    _data->d_cellOcc.assign(nCells,0);
    gpuErrchk( cudaPeekAtLastError() );
    countCellOccupancyD<<<nBlocks, nThreads>>>((uint *)thrust::raw_pointer_cast(_data->d_cellOcc.data()),
                                               (uint *)thrust::raw_pointer_cast(_data->d_hash.data()),
                                               nCells,
                                               _num_points);
    gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    thrust::exclusive_scan(_data->d_cellOcc.begin(),_data->d_cellOcc.end(),_data->d_scatterAdd.begin());
    gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
}

void viscosity(unsigned int _N,
               unsigned int _gridRes,
               float _iRadius,
               float _timestep,
               ParticlesData * _data)
{

    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;

    viscosityD<<<nBlocks,nThreads>>>(_N,
                                    _gridRes,
                                    _iRadius,
                                    _timestep,
                                    (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                    (float *)thrust::raw_pointer_cast(_data->d_Py.data()),
                                    (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                    (float *)thrust::raw_pointer_cast(_data->d_Vy.data()),
                                    (uint *)thrust::raw_pointer_cast(_data->d_hash.data()),
                                    (uint *)thrust::raw_pointer_cast(_data->d_cellOcc.data()),
                                    (uint *)thrust::raw_pointer_cast(_data->d_scatterAdd.data()));
    gpuErrchk( cudaPeekAtLastError() );
}

void integrate(unsigned int _N,
               float _timestep,
               ParticlesData * _data)
{
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;

    integrateD<<<nBlocks,nThreads>>>(_N,
                                    _timestep,
                                    (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                    (float *)thrust::raw_pointer_cast(_data->d_Py.data()),
                                    (float *)thrust::raw_pointer_cast(_data->d_prevPx.data()),
                                    (float *)thrust::raw_pointer_cast(_data->d_prevPy.data()),
                                    (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                    (float *)thrust::raw_pointer_cast(_data->d_Vy.data()));

    boundaries<<<nBlocks,nThreads>>>(_N,
                                     (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                     (float *)thrust::raw_pointer_cast(_data->d_Py.data()));
}

void density(unsigned int _N,
              unsigned int _gridRes,
              float _iRadius,
              float _timestep,
              ParticlesData * _data)
{

    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;

    densityD<<<nBlocks,nThreads>>>(_N,
                                  _gridRes,
                                  _iRadius,
                                  _timestep,
                                  (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                  (float *)thrust::raw_pointer_cast(_data->d_Py.data()),
                                  (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                  (float *)thrust::raw_pointer_cast(_data->d_Vy.data()),
                                  (uint *)thrust::raw_pointer_cast(_data->d_hash.data()),
                                  (uint *)thrust::raw_pointer_cast(_data->d_cellOcc.data()),
                                  (uint *)thrust::raw_pointer_cast(_data->d_scatterAdd.data()));

    boundaries<<<nBlocks,nThreads>>>(_N,
                                     (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                     (float *)thrust::raw_pointer_cast(_data->d_Py.data()));
}

void updateVelocity(unsigned int _N,
                    float _timestep,
                    ParticlesData * _data)
{
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;
    updateVelocityD<<<nBlocks,nThreads>>>(_N,
                                          _timestep,
                                          (float *)thrust::raw_pointer_cast(_data->d_Px.data()),
                                          (float *)thrust::raw_pointer_cast(_data->d_Py.data()),
                                          (float *)thrust::raw_pointer_cast(_data->d_prevPx.data()),
                                          (float *)thrust::raw_pointer_cast(_data->d_prevPy.data()),
                                          (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                          (float *)thrust::raw_pointer_cast(_data->d_Vy.data()));
}

void addGravity(unsigned int _N,
                ParticlesData * _data)
{
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;
    addGravityD<<<nBlocks,nThreads>>>(_N,
                                      (float *)thrust::raw_pointer_cast(_data->d_Vx.data()),
                                      (float *)thrust::raw_pointer_cast(_data->d_Vy.data()));
}
//}

