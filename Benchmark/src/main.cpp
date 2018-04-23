#include <QCoreApplication>
#include <benchmark/benchmark.h>
#include "World_cpu.h"
#include "World_gpu.h"

//---------------------------------------------------------

static void CPU_worldCreation( benchmark::State& state )
{
    for( auto _ : state)
        benchmark::DoNotOptimize( WorldCPU() );
}
BENCHMARK(CPU_worldCreation);

static void GPU_worldCreation( benchmark::State& state )
{
    for( auto _ : state)
        benchmark::DoNotOptimize( WorldGPU() );
}
BENCHMARK(GPU_worldCreation);

static void CPU_worldTenThousandCreation( benchmark::State& state )
{
    for( auto _ : state)
        benchmark::DoNotOptimize( WorldCPU(10000,0.005f,0.002f,4) );
}
BENCHMARK(CPU_worldTenThousandCreation);

static void GPU_worldTenThousandCreation( benchmark::State& state )
{
    for( auto _ : state)
        benchmark::DoNotOptimize( WorldGPU(10000,0.005f,0.002f,4) );
}
BENCHMARK(GPU_worldTenThousandCreation);

static void CPU_worldMillionCreation( benchmark::State& state )
{
    for( auto _ : state)
        benchmark::DoNotOptimize( WorldCPU(1000000,0.0005f,0.002f,4) );
}
BENCHMARK(CPU_worldMillionCreation);

static void GPU_worldMillionCreation( benchmark::State& state )
{
    for( auto _ : state)
        benchmark::DoNotOptimize( WorldGPU(1000000,0.0005f,0.002f,4) );
}
BENCHMARK(GPU_worldMillionCreation);

//-----------------------HUNDRED SIMULATE----------------------------------

static void CPU_simulateHundred( benchmark::State& state )
{
    WorldCPU myWorld(100,0.05f,0.02f,4);

    for ( auto _ : state )
    {
      myWorld.simulate(1);
    }
}
BENCHMARK(CPU_simulateHundred);

static void GPU_simulateHundred( benchmark::State& state )
{
    WorldGPU myWorld(100,0.05f,0.02f,4);

    for ( auto _ : state )
    {
      myWorld.simulate(1);
    }
}
BENCHMARK(GPU_simulateHundred);

//-----------------------TEN THOUSAND SIMULATE------------------------------

static void CPU_simulateTenThosand( benchmark::State& state )
{
    WorldCPU myWorld(10000,0.005f,0.002f,4);

    for ( auto _ : state )
    {
      myWorld.simulate(1);
    }
}
BENCHMARK(CPU_simulateTenThosand);

static void GPU_simulateTenThosand( benchmark::State& state )
{
    WorldGPU myWorld(10000,0.005f,0.002f,4);

    for ( auto _ : state )
    {
      myWorld.simulate(1);
    }
}
BENCHMARK(GPU_simulateTenThosand);


BENCHMARK_MAIN();
