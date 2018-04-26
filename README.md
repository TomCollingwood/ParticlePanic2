# ParticlePanic2
CUDA implementation of SPH particles

## CPU library (PP2_cpu)

The CPU library uses nearest neighbour optimization but in serial. No multithreading or parallelism. The parameterized constructor has four inputs.

* Number of particles
* Interaction radius
* Timestep
* Grid resolution

Call .simulate(1) to simulate a frame. Increase the input integer to increase number of substeps. Each substep has timestep that you input with parameterized constructor.

.dumpToGeo(int i) will dump the particle data into a Houdini geo file into folder

## GPU library (PP2_gpu)

The GPU library uses nearest neigbour optimization using thrust. Then there are kernels for each stage of the SPH simulation. Has a parameterized constructor like PP2_cpu.

* Gravity is added to each particles velocity via a kernel
* Viscosity is calculated and added to particles velocities via a kernel
* Spatial hashing is calculated
  * A kernel calculates the hash index for each particle and puts it in a thrust::device\_vector d\_hash in same order as the particles.
  * Then we sort the particles device\_vectors by hash value.
  * A kernel counts the cell occupancy of each cell and places it in a device\_vector d\_cellOcc . Each cell has their own hash value.
  * thrust::exclusive\_scan is carried out on d\_cellOcc and the result is d\_scatterAdd. This gives us the index for the particles for a given cell.
* Integrate the velocity and position via a kernel.
* A kernel calculates density for each particle and moves the particles accordingly.
* A kernel updates velocity based on previous position and current position.

## GPU application

To build type "make" into terminal when in PP2GPUMain directory. Then you run the file. There are four arguments you can pass into the application. In order they are:

* Number of particles
* Number of frames
* Interaction radius
* Timestep

## Comparison of times

The times below were recorded on a Intel® Xeon(R) CPU E5-1650 v3 @ 3.50GHz × 12 processor, a Quadro K2200 graphics card and 32GB of RAM.

Number of Particles | CPU library Time per Frame (seconds) | GPU library Time per Frame (seconds) | GPU App time per frame (seconds)
------------------- | ------------------------------------ | ------------------------------------ | --------------------------------
100 		| ~0.0022 	| ~0.014	| ~0.0012
10,000 		| ~13.0 	| ~0.145 	| ~0.055
100,000 	| ~1001.97	| ~16.23	| ~7.05

## Benchmark

Benchmark | Time | CPU | Iterations
--------- | ---- | --- | ----------
CPU_worldCreation            |      19154 ns |      19155 ns |     29709
GPU_worldCreation            |    2845229 ns |    2845284 ns |       232
CPU_worldTenThousandCreation |    2019773 ns |    2019932 ns |       342
GPU_worldTenThousandCreation |    7833039 ns |    7833436 ns |        88
CPU_worldMillionCreation     |  209919534 ns |  209941143 ns |         3
GPU_worldMillionCreation     |  309583895 ns |  309556235 ns |         2
CPU_simulateHundred          |    1467880 ns |    1467941 ns |       404
GPU_simulateHundred          |   14201930 ns |   14201589 ns |        46
CPU_simulateTenThosand       | 7586408742 ns | 7586935837 ns |         1
GPU_simulateTenThosand       |  338824467 ns |  338829504 ns |         4

## Issues

Sometimes there is an illegal memory access error that is asserted. This can be fixed by restarting your computer and running it again immediately.



