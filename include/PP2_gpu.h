#ifndef _PP2GPU_H_
#define _PP2GPU_H_

#include <iostream>


#include <vector>

// Needed for output functions within the kernel
#include <stdio.h>

namespace PP2_GPU {

    float * d_Px_ptr;
    float * d_Py_ptr;
    float * d_prevPx_ptr;
    float * d_prevPy_ptr;
    float * d_Vx_ptr;
    float * d_Vy_ptr;

    int m_numPoints = 0;
    int m_gridResolution = 4;
    float m_interactionRadius = 0.05f;
    float m_timestep = 1.0f;
    bool m_started = false;


    unsigned int * d_hash_ptr;
    unsigned int * d_cellOcc_ptr;
    unsigned int * d_scatterAdd_ptr;

    void initData();

    void hashOccSort();

    void addParticle(float P_x, float P_y, float V_x, float V_y);

    void castPointers();

    void simulate();

    int getNumPoints();

    void clearMem();

    //--------------- INPUTS-------------------------

    bool m_rain = true;
    bool m_gravity;

}

#endif //_RAND_GPU_H
