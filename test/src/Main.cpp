///
///  @file    Main.cpp
///  @brief   contains the main method - creates window and runs the simulation
///  @author  Richard Southern & Thomas Collingwood

#ifdef _WIN32
#include <windows.h>
#endif

#include <iostream>
#include <sys/time.h>

#include "World_cpu.h"
//#include "World_gpu.h"

// Our World, which will store all the GL stuff
WorldCPU *m_worldCPU = NULL;

/// This function was originally written by Richard Southern in his Cube workshop
int main( int argc, char* args[] ) {

    m_worldCPU = new WorldCPU();

    int num_frames = 60;

    struct timeval tim;
            double t1, t2, t3;
            gettimeofday(&tim, NULL);

    for(int i = 1; i<num_frames; ++i)
    {
        gettimeofday(&tim, NULL);
        t1=tim.tv_sec+(tim.tv_usec/1000000.0);
        m_worldCPU->simulate();
        gettimeofday(&tim, NULL);
        t2=tim.tv_sec+(tim.tv_usec/1000000.0);
        t3 = t2-t1;
        printf("Frame %d took %f seconds\n",i,t3);
        m_worldCPU->dumpToGeo(i);
    }

    return EXIT_SUCCESS;
}
/// end of function
