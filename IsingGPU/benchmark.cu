#include "Ising_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>


/*
        ##################################################################################################
        #                                                                                                #
        #    before compiling the benchmark: COMMENT THE MACROS AND THREAD GEOMETRYin the header file    #
        #                                                                                                #
        ##################################################################################################
*/

int main() {

    // n of benchamarks = n of different lattice dimensions
    int numBenchmarks = 7;
    float *tim = (float *)malloc(numBenchmarks * sizeof(float));
    int   *wid = (int *)malloc(numBenchmarks * sizeof(int));

    // memory allocation for benchmarks 
    float *elapsedMagnetizationArr = (float *)malloc(numBenchmarks * sizeof(float));
    float *elapsedEnergyArr        = (float *)malloc(numBenchmarks * sizeof(float));
    float *elapsedMontecarloArr    = (float *)malloc(numBenchmarks * sizeof(float));

    float *bwMagnetizationArr      = (float *)malloc(numBenchmarks * sizeof(float));
    float *bwEnergyArr             = (float *)malloc(numBenchmarks * sizeof(float));
    float *bwMontecarloArr         = (float *)malloc(numBenchmarks * sizeof(float));

    // benchamrk index
    int index = 0;

    // loop over different WIDTH (power of 2)
    for (int w = 16; w <= 1024; w *= 2) {

        clock_t start_time = clock();

        // benchmark time initialization for the kernels
        float total_elapsed_magnetization = 0.0f;
        float total_elapsed_energy = 0.0f;
        float total_elapsed_montecarlo = 0.0f;

        wid[index] = w;

        // Simulation parameters
        int WIDTH = w;
        int SHAPE = WIDTH * WIDTH;
        float J = 1.0f;
        float H = -0.0f;
        float T = 2.2f;
        int N_STEPS = 1000;
        int SAVE_INTERVAL = 100;

        // kernel eometry given the lattice geometry
        int THREADS_PER_BLOCK_X = 16;
        int THREADS_PER_BLOCK_Y = 16;
        int THREADS_PER_BLOCK = THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y;
        int N_BLOCKS_X = (WIDTH + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X;
        int N_BLOCKS_Y = (WIDTH + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y;
        int N_BLOCKS = N_BLOCKS_X * N_BLOCKS_Y;

        // reduction kernels geometry 
        int REDUCTION_THREADS_PER_BLOCK = 256;
        int REDUCTION_N_BLOCKS = (SHAPE + (2 * REDUCTION_THREADS_PER_BLOCK - 1)) / (2 * REDUCTION_THREADS_PER_BLOCK);

        int n_save = N_STEPS / SAVE_INTERVAL;

        // host memory alllocation
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

        // seed and first state initializaiton
        srand(time(NULL));
        InitializeSeed(h_InitialSeed, SHAPE);
        InitializeRandomState(h_InitialState, SHAPE);

        // copy the initialized data into the gpu
        cudaMemcpy(d_CurrentSeed, h_InitialSeed, SHAPE * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_NewSeed, h_InitialSeed, SHAPE * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_CurrentState, h_InitialState, SHAPE * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_StateHistory, d_CurrentState, SHAPE * sizeof(int), cudaMemcpyDeviceToDevice);

        // thread geometry
        dim3 blockSize(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
        dim3 gridSize(N_BLOCKS_X, N_BLOCKS_Y);

        // Cuda events for benchmarks
        cudaEvent_t start_energy, stop_energy;
        cudaEvent_t start_magnetization, stop_magnetization;
        cudaEvent_t start_montecarlo_a, stop_montecarlo_a;
        cudaEvent_t start_montecarlo_b, stop_montecarlo_b;
        cudaEventCreate(&start_energy);
        cudaEventCreate(&stop_energy);
        cudaEventCreate(&start_magnetization);
        cudaEventCreate(&stop_magnetization);
        cudaEventCreate(&start_montecarlo_a);
        cudaEventCreate(&stop_montecarlo_a);
        cudaEventCreate(&start_montecarlo_b);
        cudaEventCreate(&stop_montecarlo_b);

        // Montecarlo simulation
        // Kernels and time benchmarks 
        for (int step = 1; step < N_STEPS; step++) {

            cudaEventRecord(start_montecarlo_a);
            MonteCarloStepCheckerboard<<<gridSize, blockSize>>>(d_CurrentState, d_NewState,
                                                                d_CurrentSeed, d_NewSeed, T, H, J, WIDTH, 0);
            cudaEventRecord(stop_montecarlo_a);
            cudaEventSynchronize(stop_montecarlo_a);
            cudaDeviceSynchronize();

            int *temp_state = d_CurrentState;
            d_CurrentState = d_NewState;
            d_NewState = temp_state;
            int *temp_seed = d_CurrentSeed;
            d_CurrentSeed = d_NewSeed;
            d_NewSeed = temp_seed;

            cudaEventRecord(start_montecarlo_b);
            MonteCarloStepCheckerboard<<<gridSize, blockSize>>>(d_CurrentState, d_NewState,
                                                                d_CurrentSeed, d_NewSeed, T, H, J, WIDTH, 1);
            cudaEventRecord(stop_montecarlo_b);
            cudaEventSynchronize(stop_montecarlo_b);
            cudaDeviceSynchronize();

            temp_state = d_CurrentState;
            d_CurrentState = d_NewState;
            d_NewState = temp_state;
            temp_seed = d_CurrentSeed;
            d_CurrentSeed = d_NewSeed;
            d_NewSeed = temp_seed;

            if (step % SAVE_INTERVAL == 0) {

                int saved_index = step / SAVE_INTERVAL;

                cudaEventRecord(start_magnetization);
                reduceMagnetization<<<REDUCTION_N_BLOCKS, REDUCTION_THREADS_PER_BLOCK, 
                                       REDUCTION_THREADS_PER_BLOCK * sizeof(float)>>>(d_CurrentState, d_BlockMagnetization, SHAPE);
                cudaEventRecord(stop_magnetization);
                cudaEventSynchronize(stop_magnetization);   
                cudaDeviceSynchronize();

                cudaEventRecord(start_energy);                
                reduceEnergy<<<REDUCTION_N_BLOCKS, REDUCTION_THREADS_PER_BLOCK, 
                               REDUCTION_THREADS_PER_BLOCK * sizeof(float)>>>(d_CurrentState, d_BlockEnergy, WIDTH, SHAPE, H, J);
                cudaEventRecord(stop_energy);
                cudaEventSynchronize(stop_energy);   
                cudaDeviceSynchronize();

                cudaMemcpy(d_StateHistory + saved_index * SHAPE, d_CurrentState,
                           SHAPE * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_BlockMagnetizationHistory + saved_index * N_BLOCKS, d_BlockMagnetization,
                           N_BLOCKS * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_BlockEnergyHistory + saved_index * N_BLOCKS, d_BlockEnergy,
                           N_BLOCKS * sizeof(float), cudaMemcpyDeviceToDevice);

                // time benchmark for reduction kernels
                float elapsed_magnetization;
                cudaEventElapsedTime(&elapsed_magnetization, start_magnetization, stop_magnetization);
                total_elapsed_magnetization += elapsed_magnetization;

                float elapsed_energy;
                cudaEventElapsedTime(&elapsed_energy, start_energy, stop_energy);
                total_elapsed_energy += elapsed_energy;
            }

            // time benchmark for montecarlo kernel
            float elapsed_montecarlo_a, elapsed_montecarlo_b;
            cudaEventElapsedTime(&elapsed_montecarlo_a, start_montecarlo_a, stop_montecarlo_a);
            cudaEventElapsedTime(&elapsed_montecarlo_b, start_montecarlo_b, stop_montecarlo_b);
            total_elapsed_montecarlo += (elapsed_montecarlo_a + elapsed_montecarlo_b);

        }

        cudaEventDestroy(start_energy);
        cudaEventDestroy(stop_energy);
        cudaEventDestroy(start_magnetization);
        cudaEventDestroy(stop_magnetization);
        cudaEventDestroy(start_montecarlo_a);
        cudaEventDestroy(stop_montecarlo_a);
        cudaEventDestroy(start_montecarlo_b);
        cudaEventDestroy(stop_montecarlo_b);
        
        // mean of the measured time
        total_elapsed_magnetization /= n_save;
        total_elapsed_energy /= n_save;
        total_elapsed_montecarlo /= N_STEPS;

        cudaMemcpy(h_StateHistory, d_StateHistory, n_save * SHAPE * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_BlockMagnetizationHistory, d_BlockMagnetizationHistory, n_save * N_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_BlockEnergyHistory, d_BlockEnergyHistory, n_save * N_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        computeMagnetization(h_BlockMagnetizationHistory, h_MagnetizationHistory, N_BLOCKS, n_save);
        computeEnergy(h_BlockEnergyHistory, h_EnergyHistory, N_BLOCKS, n_save);

        // end of the simulation 
        clock_t end_time = clock();
        float sim_time = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;
        tim[index] = sim_time;

        // BANDWITH EVALUATION 
        //      -Energy: (3 sites) input memory access
        //      -Magnetization: (2 sites) input memory access memory access 
        //      -Montecarlo: (4 nearest neighbor + 1 site + 1 seed) input memory acces + (1 site + 1 seed) output mmemory access
        float energy_bandwidth = (REDUCTION_N_BLOCKS * REDUCTION_THREADS_PER_BLOCK * 3.0 * sizeof(float)) / (total_elapsed_energy * 1e6);
        float magnetization_bandwidth = (REDUCTION_N_BLOCKS * REDUCTION_THREADS_PER_BLOCK * 2.0 * sizeof(float)) / (total_elapsed_magnetization * 1e6);
        float montecarlo_bandwidth = (N_BLOCKS * THREADS_PER_BLOCK * 8.0 * sizeof(float)) / (total_elapsed_montecarlo * 1e6);

        // saving final result
        elapsedMagnetizationArr[index] = total_elapsed_magnetization;
        elapsedEnergyArr[index]        = total_elapsed_energy;
        elapsedMontecarloArr[index]    = total_elapsed_montecarlo;
        bwMagnetizationArr[index]      = magnetization_bandwidth;
        bwEnergyArr[index]             = energy_bandwidth;
        bwMontecarloArr[index]         = montecarlo_bandwidth;

        // print out
        printf("\nBenchmark for width = %d completed in %.4f seconds\n", WIDTH, sim_time);
        printf("Kernels elapsed time / Bandwidth:\n");
        printf("  - reduceMagnetization: %.3f ms / %.3f GB/s\n", total_elapsed_magnetization, magnetization_bandwidth);
        printf("  - reduceEnergy: %.3f ms / %.3f GB/s\n", total_elapsed_energy, energy_bandwidth);
        printf("  - MonteCarloStepCheckerboard: %.3f ms / %.3f GB/s\n", total_elapsed_montecarlo, montecarlo_bandwidth);

       
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

        index++;
    } // end of the benchmarks

    // writing the results on csv file 
    FILE *fp = fopen("benchmark_results_1000.csv", "w");
    fprintf(fp, "Width,CompleteTime,reduceMagnetization_elapsed_ms,reduceMagnetization_BW_GBs,reduceEnergy_elapsed_ms,reduceEnergy_BW_GBs,MonteCarloStepCheckerboard_elapsed_ms,MonteCarloStepCheckerboard_BW_GBs\n");
    for (int i = 0; i < numBenchmarks; i++) {
        fprintf(fp, "%d,%.7f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                wid[i], tim[i],
                elapsedMagnetizationArr[i], bwMagnetizationArr[i],
                elapsedEnergyArr[i], bwEnergyArr[i],
                elapsedMontecarloArr[i], bwMontecarloArr[i]);
    }
    fclose(fp);


    free(tim);
    free(wid);
    free(elapsedMagnetizationArr);
    free(elapsedEnergyArr);
    free(elapsedMontecarloArr);
    free(bwMagnetizationArr);
    free(bwEnergyArr);
    free(bwMontecarloArr);

    return 0;
}