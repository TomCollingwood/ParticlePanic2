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

#include <sstream>
#include <fstream>

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

ParticlesData * initializeParticlesData(const int _numPoints, const int _gridRes)
{
    ParticlesData * _data = new ParticlesData();

    int nCells = _gridRes*_gridRes;

    //gpuErrchk( cudaPeekAtLastError() );

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

    _data->d_Px = thrust::device_vector<float>(_numPoints,0.0f);
    _data->d_Py = thrust::device_vector<float>(_numPoints,0.0f);
    _data->d_prevPx = thrust::device_vector<float>(_numPoints,0.0f);
    _data->d_prevPy = thrust::device_vector<float>(_numPoints,0.0f);

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

void deleteData(ParticlesData * io_data)
{
    delete io_data;
}

void dumpToGeoCUDA(ParticlesData *_data,
                   const uint _cnt)
{
    char fname[150];

    std::sprintf(fname,"geo/SPH_GPU.%03d.geo",_cnt);
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

void hashOccSort(int _num_points,
                 int _gridRes,
                 ParticlesData * io_data)
{
    if(_num_points==0) return;
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _num_points / nThreads + 1;
    unsigned int nCells = _gridRes*_gridRes;

    //gpuErrchk( cudaPeekAtLastError() );

    pointHash2D<<<nBlocks, nThreads>>>(thrust::raw_pointer_cast(io_data->d_hash.data()),
                                       thrust::raw_pointer_cast(io_data->d_Px.data()),
                                       thrust::raw_pointer_cast(io_data->d_Py.data()),
                                       _num_points,
                                       _gridRes);

    //gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    //if(cudaSuccess==cudaThreadSynchronize()) std::cout<<"it works"<<std::endl;

    //gpuErrchk( cudaPeekAtLastError() ); // stops here
    auto tuple = thrust::make_tuple( io_data->d_Px.begin(), io_data->d_Py.begin(), io_data->d_Vx.begin(), io_data->d_Vy.begin(), io_data->d_prevPx.begin(), io_data->d_prevPy.begin());
    //gpuErrchk( cudaPeekAtLastError() );
    auto zippy = thrust::make_zip_iterator(tuple);
    //gpuErrchk( cudaPeekAtLastError() );
    thrust::sort_by_key(io_data->d_hash.begin(), io_data->d_hash.end(), zippy); // bad alloc here

    //gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    io_data->d_cellOcc.assign(nCells,0);
    //gpuErrchk( cudaPeekAtLastError() );
    countCellOccupancyD<<<nBlocks, nThreads>>>((uint *)thrust::raw_pointer_cast(io_data->d_cellOcc.data()),
                                               (uint *)thrust::raw_pointer_cast(io_data->d_hash.data()),
                                               nCells,
                                               _num_points);
    //gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    //gpuErrchk( cudaPeekAtLastError() );
    thrust::exclusive_scan(io_data->d_cellOcc.begin(),io_data->d_cellOcc.end(),io_data->d_scatterAdd.begin());
    //gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
}

void viscosity(unsigned int _N,
               unsigned int _gridRes,
               float _iRadius,
               float _timestep,
               ParticlesData * io_data)
{

    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;

    viscosityD<<<nBlocks,nThreads>>>(_N,
                                     _gridRes,
                                     _iRadius,
                                     _timestep,
                                     (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Py.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Vy.data()),
                                     (uint *)thrust::raw_pointer_cast(io_data->d_hash.data()),
                                     (uint *)thrust::raw_pointer_cast(io_data->d_cellOcc.data()),
                                     (uint *)thrust::raw_pointer_cast(io_data->d_scatterAdd.data()));
    //gpuErrchk( cudaPeekAtLastError() );
}

void integrate(unsigned int _N,
               float _timestep,
               ParticlesData * io_data)
{
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;

    integrateD<<<nBlocks,nThreads>>>(_N,
                                     _timestep,
                                     (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Py.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_prevPx.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_prevPy.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Vy.data()));

    boundaries<<<nBlocks,nThreads>>>(_N,
                                     (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Py.data()));
}

void density(unsigned int _N,
             unsigned int _gridRes,
             float _iRadius,
             float _timestep,
             ParticlesData * io_data)
{

    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;

    densityD<<<nBlocks,nThreads>>>(_N,
                                   _gridRes,
                                   _iRadius,
                                   _timestep,
                                   (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                   (float *)thrust::raw_pointer_cast(io_data->d_Py.data()),
                                   (float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),
                                   (float *)thrust::raw_pointer_cast(io_data->d_Vy.data()),
                                   (uint *)thrust::raw_pointer_cast(io_data->d_hash.data()),
                                   (uint *)thrust::raw_pointer_cast(io_data->d_cellOcc.data()),
                                   (uint *)thrust::raw_pointer_cast(io_data->d_scatterAdd.data()));

    boundaries<<<nBlocks,nThreads>>>(_N,
                                     (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Py.data()));
}

void updateVelocity(unsigned int _N,
                    float _timestep,
                    ParticlesData * io_data)
{
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;
    updateVelocityD<<<nBlocks,nThreads>>>(_N,
                                          _timestep,
                                          (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_Py.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_prevPx.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_prevPy.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_Vy.data()));
}

void addGravity(unsigned int _N,
                ParticlesData * io_data)
{
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;
    addGravityD<<<nBlocks,nThreads>>>(_N,
                                      (float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),
                                      (float *)thrust::raw_pointer_cast(io_data->d_Vy.data()));
}

void simulateD(unsigned int _N,
               unsigned int _gridRes,
               float _iRadius,
               float _timestep,
               ParticlesData * io_data)
{
    unsigned int nThreads = 1024;
    unsigned int nBlocks = _N / nThreads + 1;
    //-------------------------GRAVITY----------------------------------------
    addGravityD<<<nBlocks, nThreads>>>(_N,(float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),(float *)thrust::raw_pointer_cast(io_data->d_Vy.data()));
    //gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();

    //-------------------------VISCOSITY--------------------------------------
    viscosityD<<<nBlocks, nThreads>>>(_N,
                                      _gridRes,
                                      _iRadius,
                                      _timestep,
                                      (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                      (float *)thrust::raw_pointer_cast(io_data->d_Py.data()),
                                      (float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),
                                      (float *)thrust::raw_pointer_cast(io_data->d_Vy.data()),
                                      (uint *)thrust::raw_pointer_cast(io_data->d_hash.data()),
                                      (uint *)thrust::raw_pointer_cast(io_data->d_cellOcc.data()),
                                      (uint *)thrust::raw_pointer_cast(io_data->d_scatterAdd.data()));
    //gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();

    //--------------------------INTEGRATE-------------------------------------
    integrateD<<<nBlocks,nThreads>>>(_N,
                                     _timestep,
                                     (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Py.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_prevPx.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_prevPy.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Vy.data()));
    //gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    boundaries<<<nBlocks,nThreads>>>(_N,
                                     (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Py.data()));
    cudaThreadSynchronize();

    //---------------------------HASHING---------------------------------------
    pointHash2D<<<nBlocks, nThreads>>>((uint *)thrust::raw_pointer_cast(io_data->d_hash.data()),
                                       (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                       (float *)thrust::raw_pointer_cast(io_data->d_Py.data()),
                                       _N,
                                       _gridRes);
    cudaThreadSynchronize();

    // Now we can sort our points to ensure that points in the same grid cells occupy contiguous memory
    thrust::sort_by_key(io_data->d_hash.begin(), io_data->d_hash.end(),
                        thrust::make_zip_iterator(
                            thrust::make_tuple( io_data->d_Px.begin(),
                                                io_data->d_Py.begin(),
                                                io_data->d_prevPx.begin(),
                                                io_data->d_prevPy.begin(),
                                                io_data->d_Vx.begin(),
                                                io_data->d_Vy.begin())));
    cudaThreadSynchronize();

    io_data->d_cellOcc.assign(_gridRes*_gridRes,0);
    // Now we can count the number of points in each grid cell
    countCellOccupancyD<<<nBlocks, nThreads>>>((uint *)thrust::raw_pointer_cast(io_data->d_cellOcc.data()),
                                               (uint *)thrust::raw_pointer_cast(io_data->d_hash.data()),
                                               io_data->d_cellOcc.size(), io_data->d_hash.size());

    // DONE CORRECTLY
    cudaThreadSynchronize();
    io_data->d_scatterAdd.resize(_gridRes*_gridRes,0);
    thrust::exclusive_scan(io_data->d_cellOcc.begin(),io_data->d_cellOcc.end(),io_data->d_scatterAdd.begin());
    cudaThreadSynchronize();


    // Only dump the debugging information if we have a manageable number of points.
    // if (_N <= 100) {
    //     std::cout << "\n";
    //     thrust::copy(d_hash.begin(), d_hash.end(), std::ostream_iterator<unsigned int>(std::cout, " "));
    //     std::cout << "\n";
    //     thrust::copy(d_cellOcc.begin(), d_cellOcc.end(), std::ostream_iterator<unsigned int>(std::cout, " "));
    //     std::cout << "\n";
    //     thrust::copy(d_scatterAdd.begin(), d_scatterAdd.end(), std::ostream_iterator<unsigned int>(std::cout, " "));
    //     std::cout << "\n";
    // }


    //---------------------------DOUBLE DENSITY------------------------------

    //gpuErrchk( cudaPeekAtLastError() );
    densityD<<<nBlocks,nThreads>>>(_N,
                                   _gridRes,
                                   _iRadius,
                                   _timestep,
                                   (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                   (float *)thrust::raw_pointer_cast(io_data->d_Py.data()),
                                   (float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),
                                   (float *)thrust::raw_pointer_cast(io_data->d_Vy.data()),
                                   (uint *)thrust::raw_pointer_cast(io_data->d_hash.data()),
                                   (uint *)thrust::raw_pointer_cast(io_data->d_cellOcc.data()),
                                   (uint *)thrust::raw_pointer_cast(io_data->d_scatterAdd.data()));

    //gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    boundaries<<<nBlocks,nThreads>>>(_N,
                                     (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                     (float *)thrust::raw_pointer_cast(io_data->d_Py.data()));
    cudaThreadSynchronize();
    //gpuErrchk( cudaPeekAtLastError() );

    //---------------------------UPDATE VELOCITY-------------------------------
    updateVelocityD<<<nBlocks,nThreads>>>(_N,
                                          _timestep,
                                          (float *)thrust::raw_pointer_cast(io_data->d_Px.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_Py.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_prevPx.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_prevPy.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_Vx.data()),
                                          (float *)thrust::raw_pointer_cast(io_data->d_Vy.data()));
    //gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();

}

//}

