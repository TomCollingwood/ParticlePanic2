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

Number of Particles | CPU library Time per Frame (seconds) | GPU library Time per Frame (seconds) | GPU App time per frame (seconds)
------------------- | ------------------------------------ | ------------------------------------ | --------------------------------
100 		| ~0.0022 	|		|
10,000 		| ~13.0 	| ~0.145 	|
100,000 	| ~1001.97	| 		|
