#include "Ising_cuda.h"
#include <stdio.h>
#include <time.h>

/*
        ################################################################################################
        #                                                                                              #
        #    Before compiling the main program, make sure to UNCOMMENT THE MACROS in the header file   #
        #                                                                                              #
        ################################################################################################
*/

// Mainfunction to compute a montecarlo simulation of a 2D ising model 
int main() {

    // n of snapshots of the simulation 
    int n_save = N_STEPS / SAVE_INTERVAL;

    // host memory allocation 
    int   *h_InitialState = (int *)malloc(SHAPE * sizeof(int));
    int   *h_StateHistory = (int *)malloc(n_save * SHAPE * sizeof(int));
    int   *h_InitialSeed  = (int *)malloc(SHAPE * sizeof(int));
    float *h_BlockMagnetizationHistory = (float *)malloc(N_BLOCKS * n_save * sizeof(float));
    float *h_MagnetizationHistory = (float *)malloc(n_save * sizeof(float));
    float *h_BlockEnergyHistory = (float *)malloc(N_BLOCKS * n_save * sizeof(float));
    float *h_EnergyHistory = (float *)malloc(n_save * sizeof(float));

    // device memory allocation 
    int   *d_CurrentState, *d_NewState, *d_CurrentSeed, *d_NewSeed, *d_StateHistory;
    float *d_BlockMagnetization, *d_BlockMagnetizationHistory;
    float *d_BlockEnergy, *d_BlockEnergyHistory;

    cudaMalloc((void**)&d_CurrentState, SHAPE * sizeof(int));
    cudaMalloc((void**)&d_NewState, SHAPE * sizeof(int));
    cudaMalloc((void**)&d_CurrentSeed, SHAPE * sizeof(int));
    cudaMalloc((void**)&d_NewSeed, SHAPE * sizeof(int));
    cudaMalloc((void**)&d_StateHistory, n_save * SHAPE * sizeof(int));
    cudaMalloc((void**)&d_BlockMagnetization, N_BLOCKS * sizeof(float));
    cudaMalloc((void**)&d_BlockMagnetizationHistory, n_save * N_BLOCKS * sizeof(float));
    cudaMalloc((void**)&d_BlockEnergy, N_BLOCKS * sizeof(float));
    cudaMalloc((void**)&d_BlockEnergyHistory, n_save * N_BLOCKS * sizeof(float));

    // random number initialization  
    srand(time(NULL));
    InitializeSeed(h_InitialSeed, SHAPE);

    // first state initialization 
    InitializeRandomState(h_InitialState, SHAPE);

    // first state and seeds passed to the gpu 
    cudaMemcpy(d_CurrentSeed, h_InitialSeed, SHAPE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_NewSeed, h_InitialSeed, SHAPE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CurrentState, h_InitialState, SHAPE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_StateHistory, d_CurrentState, SHAPE * sizeof(int), cudaMemcpyDeviceToDevice);

    // thread geometry for montecarlo 
    dim3 blockSize(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 gridSize(N_BLOCKS_X, N_BLOCKS_Y);

    clock_t start_time = clock();
    for (int step = 1; step < N_STEPS; step++) {

        // first checkerboard update result written in d_NewState memory 
        MonteCarloStepCheckerboard<<<gridSize, blockSize>>>(d_CurrentState, d_NewState, 
                                                            d_CurrentSeed, d_NewSeed, T, H, J, WIDTH, 0);
        cudaDeviceSynchronize();

        // now we swap pointers in order to have d_NewState as current state 
        int *temp_state = d_CurrentState;
        d_CurrentState = d_NewState;
        d_NewState = temp_state;
        int *temp_seed = d_CurrentSeed;
        d_CurrentSeed = d_NewSeed;
        d_NewSeed = temp_seed;

        // first checkerboard update
        MonteCarloStepCheckerboard<<<gridSize, blockSize>>>(d_CurrentState, d_NewState,
                                                            d_CurrentSeed, d_NewSeed, T, H, J, WIDTH, 1);
        cudaDeviceSynchronize();

        temp_state = d_CurrentState;
        d_CurrentState = d_NewState;
        d_NewState = temp_state;
        temp_seed = d_CurrentSeed;
        d_CurrentSeed = d_NewSeed;
        d_NewSeed = temp_seed;

        // Measure of the energy and the magnetization
        if (step % SAVE_INTERVAL == 0) {
            int saved_index = step / SAVE_INTERVAL;

            reduceMagnetization<<<REDUCTION_N_BLOCKS, REDUCTION_THREADS_PER_BLOCK,REDUCTION_THREADS_PER_BLOCK * sizeof(float)>>>(d_CurrentState, d_BlockMagnetization, SHAPE);
            reduceEnergy<<<REDUCTION_N_BLOCKS, REDUCTION_THREADS_PER_BLOCK,REDUCTION_THREADS_PER_BLOCK * sizeof(float)>>>(d_CurrentState, d_BlockEnergy, WIDTH, SHAPE, H, J);
            
            cudaMemcpy(d_StateHistory + saved_index * SHAPE, d_CurrentState,
                       SHAPE * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_BlockMagnetizationHistory + saved_index * N_BLOCKS, d_BlockMagnetization,
                       N_BLOCKS * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_BlockEnergyHistory + saved_index * N_BLOCKS, d_BlockEnergy,
                       N_BLOCKS * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    } // end of the simulation

    // copy the result into the host memory 
    cudaMemcpy(h_StateHistory, d_StateHistory, n_save * SHAPE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_BlockMagnetizationHistory, d_BlockMagnetizationHistory, n_save * N_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_BlockEnergyHistory, d_BlockEnergyHistory, n_save * N_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // final computation for the measure (outputs of the reduction kernels)
    computeMagnetization(h_BlockMagnetizationHistory, h_MagnetizationHistory, N_BLOCKS, n_save);
    computeEnergy(h_BlockEnergyHistory, h_EnergyHistory, N_BLOCKS, n_save);
    clock_t end_time = clock();

    // print out 
    printf("\n GPU MONTECARLO RUNTIME \n");
    float sim_time = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf(" %.4f secondi.\n", sim_time);

    printf("\n MAGNETIZATION EVOLUTION");
    for (int s = 0; s < n_save; s++) {
        printf("\n step: %d, Magnetization/Energy : %.4f/%.4f", s, h_MagnetizationHistory[s], h_EnergyHistory[s]);
    }

    
    printf("\n WRITING SIMULATION\n");
    SaveSimulationBinary_States("simulation_states.bin", h_StateHistory, n_save, SHAPE);
    SaveSimulationBinary_EnergyMag("simulation_magnetization.bin", h_MagnetizationHistory, n_save);
    SaveSimulationBinary_EnergyMag("simulation_energy.bin", h_EnergyHistory, n_save);
    printf("\n complete !\n");
    
    
    free(h_InitialState);
    free(h_StateHistory);
    free(h_InitialSeed);
    free(h_BlockMagnetizationHistory);
    free(h_MagnetizationHistory);
    free(h_BlockEnergyHistory);
    free(h_EnergyHistory);

    cudaFree(d_CurrentState);
    cudaFree(d_NewState);
    cudaFree(d_StateHistory);
    cudaFree(d_CurrentSeed);
    cudaFree(d_NewSeed);
    cudaFree(d_BlockMagnetization);
    cudaFree(d_BlockMagnetizationHistory);
    cudaFree(d_BlockEnergy);
    cudaFree(d_BlockEnergyHistory);

    printf("\n END ");
    return 0;
}
