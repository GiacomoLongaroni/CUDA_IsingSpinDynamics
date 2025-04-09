#ifndef ISING_H
#define ISING_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// MACROS
//#define WIDTH 512
//#define SHAPE (WIDTH * WIDTH)
//#define J 1.
//#define H -0.
//#define T 1.8
//#define N_STEPS 5000
//#define SAVE_INTERVAL 50
//
//// Montecarlo thread geometry
//#define THREADS_PER_BLOCK_X 16
//#define THREADS_PER_BLOCK_Y 16
//#define THREADS_PER_BLOCK (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
//#define N_BLOCKS_X ((WIDTH + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X)
//#define N_BLOCKS_Y ((WIDTH + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y)
//#define N_BLOCKS (N_BLOCKS_X * N_BLOCKS_Y)
//
//// Reduction thread geometry 
//#define REDUCTION_THREADS_PER_BLOCK 256
//#define REDUCTION_N_BLOCKS ((SHAPE + (2 * REDUCTION_THREADS_PER_BLOCK - 1)) / (2 * REDUCTION_THREADS_PER_BLOCK))

// functions running on cpu
void SaveSimulationBinary_States(const char *filename, const int *stateHistory, int n_save, int shape);
void SaveSimulationBinary_EnergyMag(const char *filename, const float *stateHistory, int n_save);
void InitializeRandomState(int *state, int shape);
void PrintState(int *state, int width);
void InitializeSeed(int *InitialSeed, int shape);
void computeMagnetization(float *blockHistory, float *History, int n_blocks, int n_save);
void computeEnergy(float *blockHistory, float *History, int n_blocks, int n_save);
void computeStateEnergyCPU(int *state, float *energyCPU, int width, float k, float h);

// CUDA kernels 
__global__ void reduceMagnetization(int *IsingSites, float *block_magnetization,  int shape);
__global__ void reduceEnergy(int *IsingSites, float *block_energy, int width, int shape, int h, int k);
__global__ void MonteCarloStepCheckerboard(const int *CurrentState, int *NewState,
                                             int *CurrentSeed, int *NewSeed, 
                                             float t, float h, float k, int width, int color);


#endif // ISING_H