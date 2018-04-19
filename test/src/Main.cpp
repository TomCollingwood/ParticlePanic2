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

    int num_frames = 300;

    for(int i = 0; i<num_frames; ++i)
    {
        m_worldCPU->simulate();
        m_worldCPU->dumpToObj(i);
    }

    return EXIT_SUCCESS;
}
/// end of function
