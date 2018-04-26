///
///  @file    Main.cpp
///  @brief   contains the main method - creates particle systems for GPU and CPU and tests them
///  @author  Thomas Collingwood

#include <iostream>
#include <sys/time.h>

#include "World_cpu.h"
#include "World_gpu.h"

int main( int argc, char* args[] ) {

    // HUNDRED TEST
//    WorldCPU m_worldCPU(100,0.05f,0.02f,4);
//    WorldGPU m_worldGPU(100,0.05f,0.02f,4);

    // TEN THOUSAND TEST
    WorldCPU m_worldCPU(10000,0.005f,0.002f,4);
    WorldGPU m_worldGPU(10000,0.005f,0.002f,4);

    // HUNDRED THOUSAND TEST
//    WorldCPU m_worldCPU(100000,0.0005f,0.002f,4);
//    WorldGPU m_worldGPU(100000,0.0005f,0.002f,4);

    int num_frames = 60;

    struct timeval tim;
    double t1, t2, t3;

//    for(int i = 1; i<num_frames; ++i)
//    {
//        gettimeofday(&tim, NULL);
//        t1=tim.tv_sec+(tim.tv_usec/1000000.0);

//        m_worldGPU.simulate(5);

//        gettimeofday(&tim, NULL);
//        t2=tim.tv_sec+(tim.tv_usec/1000000.0);
//        t3 = t2-t1;

//        std::cout<<"GPU Frame "<<i<<" took "<<t3<<" seconds"<<std::endl;

//        m_worldGPU.dumpToGeo(i);
//    }

//    for(int i = 1; i<num_frames; ++i)
//    {
//        gettimeofday(&tim, NULL);
//        t1=tim.tv_sec+(tim.tv_usec/1000000.0);

//        m_worldCPU.simulate(1);

//        gettimeofday(&tim, NULL);
//        t2=tim.tv_sec+(tim.tv_usec/1000000.0);
//        t3 = t2-t1;

//        std::cout<<"CPU Frame "<<i<<" took "<<t3<<" seconds"<<std::endl;

//        m_worldCPU.dumpToGeo(i);
//    }

    return EXIT_SUCCESS;
}
