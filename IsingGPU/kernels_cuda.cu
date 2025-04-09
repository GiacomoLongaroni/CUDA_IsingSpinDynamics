#include "Ising_cuda.h"
#include <math.h>

/*

    KERNELS RUNNING ON GPU USED IN THE MAIN PROGRAM AND BENCHMARKS

*/




// Kernel to perform a montecarlo update
// the montecarlo step is performed by updating half of the site (checkerboard-like map) for each kernel call 
//
//      -Input: current state, Current seeds for random numbers 
//      -Outpu: Updated state, Updated seeds for random numbers 
//
__global__ void MonteCarloStepCheckerboard(const int *CurrentState, int *NewState,
                                             int *CurrentSeed, int *NewSeed, 
                                             float t, float h, float k, int width, int color) {
    
    // parameters for the random seeds
    const unsigned int m = 2147483648u;  
    const unsigned int a = 1103515245u;
    const unsigned int c = 12345u;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < width && j < width) {
        // site index
        int idx = i * width + j;
        
        // if statement to consider half of the sites
        if ((i + j) % 2 == color) {
            // nearest neighbor index with periodic boundaries condition
            int up    = (i == 0) ? width - 1 : i - 1;
            int down  = (i == width - 1) ? 0 : i + 1;
            int left  = (j == 0) ? width - 1 : j - 1;
            int right = (j == width - 1) ? 0 : j + 1;
            
    
            int sum = CurrentState[up * width + j] +
                      CurrentState[down * width + j] +
                      CurrentState[i * width + left] +
                      CurrentState[i * width + right];

            // energy variation for the boltzmann factor
            float deltaE = 2.0f * k * CurrentState[idx] * sum + 2.0f * h * CurrentState[idx];

            // update of the seeds and pseudorandom number extraction (LGC algorithm)
            NewSeed[idx] = ((uint64_t)a * CurrentSeed[idx] + c) % m;
            float r = (float)CurrentSeed[idx] / (float)m;
            
            // metropolis criterium
            if (deltaE <= 0.0f || r < expf(-deltaE/t)) {
                NewState[idx] = -CurrentState[idx];
            } else {
                NewState[idx] = CurrentState[idx];
            }
        } else {
            NewState[idx] = CurrentState[idx];
            NewSeed[idx] = CurrentSeed[idx];
        }
    }
}



// kernel to reduce the magnetization
// M = Σs_i / size
__global__ void reduceMagnetization(int *IsingSites, float *block_magnetization,  int shape) {

    // shared memory
    extern __shared__ float partialSum[];
    unsigned int bdim = blockDim.x;
    unsigned int bx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    // global thread index
    unsigned int i = 2 * bdim * bx + tid;
    
    if (i + blockDim.x < shape && tid < shape) {

        // first sum
        // filling the shared memory with the sum of two spins 
        partialSum[tid] = IsingSites[i] + IsingSites[i + blockDim.x];
        __syncthreads();
 
 //
 //                GLOBAL-SHARED MEMORY INDEXING (eg: blockdim = 4)
 //  
 //        - each thread within a block takes two elements from the global memory: i and i + blockDim
 //                      
 //                  
 //              block: 0         block: 0     
 //              thread: 1        thread: 1 
 //              (2*4*0 + 1)=1   (2*4*0 + 1) + 4 = 5
 //              |               |   
 //              V               V
 //         0  |1  |2  |3  |4  |5  |6  |7  |8  |9  |10 |11 |12 |13 |14 |15 |     GLOBAL MEMORY  (Ising State)
 //        |_______________________________|
 //             Block 0 memory elements 
 //              
 //      
 //      
 //      result of the "loading sum"
 //              |
 //              V
 //         0  |1  |2  |3  |                                                     SHARED MEMORY (first sum )
 //
 //        
 //       - we have now filled half of the memory  [0+4,1+5,2+6,3+7, , , , ]
 //         

        // final reduction 
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partialSum[tid] += partialSum[tid + s];
            }
            __syncthreads();
        }

 //                  SHARED MEMORY REDUCTION (eg: blockdim = 4)
 //  
 //        - we define a stride s = blockDim/2 (eg: s = 2)
 //        - each thread compute the reduction between the index of his thread and the shifted one
 //                      
 //                  
 //              FIRST ITERATION (stride = 2)
 //      
 //         thread: 0       thread: 0 
 //         memory idx: 0   memory: 0+2       
 //                |        |
 //                V        V
 //               0  |1  |2  |3  |             SHARED MEMORY 
 //                    |        |     
 //                    V        V
 //            thread: 1       thread:1
 //            memory idx: 1   memory idx: 3
 //            
 //            
 //                    SECOND ITERATION (stride = 1)
 //     
 //         thread: 0       
 //         memory idx: 0      
 //                |        
 //                V        
 //               0  |1  |2  |3  |             SHARED MEMORY  
 //                    |
 //                    V
 //                thread: 0
 //                memory idx: 0 + 1        
 //
 //       - the result is stored in the 0 place 
 //        

        if (tid == 0) {
            block_magnetization[blockIdx.x] = (float)partialSum[0] / shape;
        }
    }
}




// kernel to reduce the energy of an Ising state 
// -J*ΣΣ(s_i s_j) - H Σs_i / size (s_j nearest neighbor)
/*
    same logic of the magnetization but the shared memory is filled with the energy per site
*/
__global__ void reduceEnergy(int *IsingSites, float *block_energy, int width, int shape, int h, int k) {
    extern __shared__ float partialE[];
    unsigned int bdim = blockDim.x;
    unsigned int bx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = bdim * bx + tid;

    if (i < shape) {
        int up    = (i < width) ? (i + (shape - width)) : (i - width);
        int right = ((i + 1) % width == 0) ? (i - (width - 1)) : (i + 1);
        partialE[tid] = - 4 * k * IsingSites[i] * (IsingSites[up] + IsingSites[right] + h/(2.*k));
    } else {
        partialE[tid] = 0;
    }
    
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partialE[tid] += partialE[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_energy[blockIdx.x] = (float)partialE[0] / shape;
    }
}

