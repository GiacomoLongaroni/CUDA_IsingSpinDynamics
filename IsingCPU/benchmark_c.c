#include "Ising_c.h"
#include <time.h>

/*
        ###############################################################################
        #                                                                             #
        #    before compiling the benchmark: COMMENT THE MACROS in the header file    #
        #                                                                             #
        ###############################################################################
*/


int main() {

    // n of benchamarks = n of different lattice dimensions
    int numBenchmarks = 7;

    // memory allocation for the benchmark params (times and width)
    int   *wid                  = (int *)malloc(numBenchmarks * sizeof(int));
    float *tim                  = (float *)malloc(numBenchmarks * sizeof(float));
    float *timeMagnetizationArr = (float *)malloc(numBenchmarks * sizeof(float));
    float *timeEnergyArr        = (float *)malloc(numBenchmarks * sizeof(float));
    float *timeMontecarloArr    = (float *)malloc(numBenchmarks * sizeof(float));

    // benchmark index
    int index = 0;

    // loop over different WIDTH: (power of 2)
    for (int w = 16; w <= 1024; w *= 2) {
        
        clock_t start_time = clock();
        wid[index] = w;

        // setting the benchmark params
        int WIDTH = w;
        int SHAPE = (WIDTH * WIDTH);
        int N_STEPS = 1000;
        int SAVE_INTERVAL = 100;

        // number of snapshots
        int n_save = N_STEPS / SAVE_INTERVAL;

        // Memory allocation for the simulation 
        int *state = (int *)malloc(SHAPE * sizeof(int));
        int *stateHistory = (int *)malloc(n_save * SHAPE * sizeof(int));
        float *magHistory = (float *)malloc(n_save * sizeof(float));
        float *energyHistory = (float *)malloc(n_save * sizeof(float));

        // physical constants 
        float temperature = 2.2f;  
        float J = 1.0f;            
        float H = 0.0f;     

        // random numbers initialization 
        srand((unsigned)time(NULL));

        // Initialization of the lattice
        InitializeRandomState(state, WIDTH);

        // first measure
        float mag, energy;
        ComputeMagnetization(state, &mag, WIDTH);
        computeStateEnergy(state, &energy, WIDTH, J, H);


        // initializing time for benchmark
        clock_t start_montecarlo, end_montecarlo;
        clock_t start_magnetization, end_magnetization;
        clock_t start_energy, end_energy;
        float elapsed_montecarlo;
        float elapsed_magnetization;
        float elapsed_energy;
        float elapsed_simulation;

        // Montecarlo simulation for the specific lattice dimension
        for (int step = 1; step <= N_STEPS; step++) {


            start_montecarlo = clock();
            PerformMonteCarloStep(state, WIDTH, temperature, J, H);
            end_montecarlo = clock();

            start_magnetization = clock();
            ComputeMagnetization(state, &mag, WIDTH);
            end_magnetization = clock();

            start_energy = clock();
            computeStateEnergy(state, &energy, WIDTH, J, H);
            end_energy = clock();

            if (step % SAVE_INTERVAL == 0) {
                int snapshot_index = (step / SAVE_INTERVAL) - 1;
                for (int i = 0; i < SHAPE; i++) {
                    stateHistory[snapshot_index * SHAPE + i] = state[i];
                }
                magHistory[snapshot_index] = mag;
                energyHistory[snapshot_index] = energy;

                // time measure 
                elapsed_energy = ((float)(end_energy - start_energy) * 1000.) / CLOCKS_PER_SEC;
                elapsed_magnetization = ((float)(end_magnetization - start_magnetization) *1000.) / CLOCKS_PER_SEC;
            }
            // time measure 
            elapsed_montecarlo += ((float)(end_montecarlo - start_montecarlo) * 1000.) / CLOCKS_PER_SEC;
        }

        // final result elapsed times (mean)
        clock_t end_time = clock();
        elapsed_simulation = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;
        elapsed_montecarlo /= N_STEPS;
        elapsed_energy /= n_save;
        elapsed_magnetization /= n_save;
        timeMagnetizationArr[index] = elapsed_magnetization;
        timeEnergyArr[index]        = elapsed_energy;
        timeMontecarloArr[index]    = elapsed_montecarlo;
        tim[index]                  = elapsed_simulation;
        wid[index]                  = w;

        // print out
        printf("\n");
        printf("\nBenchmark for width = %d completed in %.4f s\n", WIDTH, elapsed_simulation);
        printf("Functions elapsed time :\n");
        printf("  - Magnetization: %.4f ms", elapsed_magnetization);
        printf("  - Energy: %.4f ms", elapsed_energy);
        printf("  - MonteCarloStep: %.4f ms", elapsed_montecarlo);


        free(state);
        free(stateHistory);
        free(magHistory);
        free(energyHistory);

        index++;
    } // End of the benchmark 

    // Store the result in a csv file 
    FILE *fp = fopen("benchmark_results_cpu_1000.csv", "w");
    fprintf(fp, "Width,CompleteTime,timeMagnetization_ms, timeEnergy_ms, timeMonteCarloStep_ms\n");
    for (int i = 0; i < numBenchmarks; i++) {
        fprintf(fp, "%d,%.6f,%.4f,%.4f,%.4f\n",
                wid[i], tim[i],
                timeMagnetizationArr[i], 
                timeEnergyArr[i],
                timeMontecarloArr[i]);
    }            
    fclose(fp);


    free(tim);
    free(wid);
    free(timeMagnetizationArr);
    free(timeEnergyArr);
    free(timeMontecarloArr);

    return 0;
}

