#include "Ising_c.h"
#include <time.h>

/*
        ################################################################################################
        #                                                                                              #
        #    Before compiling the main program, make sure to UNCOMMENT THE MACROS in the header file   #
        #                                                                                              #
        ################################################################################################
*/


int main() {

    // n of snapshot we want to save 
    int n_save = N_STEPS / SAVE_INTERVAL;

    // memory allocation 
    int *state = (int *)malloc(SHAPE * sizeof(int));
    int *stateHistory = (int *)malloc(n_save * SHAPE * sizeof(int));
    float *magHistory = (float *)malloc(n_save * sizeof(float));
    float *energyHistory = (float *)malloc(n_save * sizeof(float));

    // physics simulation paramether
    float temperature = 1.8f;  
    float k = 1.0f;            
    float h_field = 0.0f;      

    // random numbers initialization
    srand((unsigned)time(NULL));

    // lattice initialization
    InitializeRandomState(state, WIDTH);

    // first state measure
    float mag, energy;
    ComputeMagnetization(state, &mag, WIDTH);
    computeStateEnergy(state, &energy, WIDTH, J, H);

    // timer initialization for benchmarks
    clock_t start_time = clock();

    // Montecarlo simulation
    for (int step = 1; step <= N_STEPS; step++) {

        // debug print
        printf("\n step: %d", step);

        // performing montecarlo step and making measures
        PerformMonteCarloStep(state, WIDTH, temperature, J, H);
        ComputeMagnetization(state, &mag, WIDTH);
        computeStateEnergy(state, &energy, WIDTH, J, H);

        // saving state and measure 
        if (step % SAVE_INTERVAL == 0) {
            int snapshot_index = (step / SAVE_INTERVAL) - 1;
            for (int i = 0; i < SHAPE; i++) {
                stateHistory[snapshot_index * SHAPE + i] = state[i];
            }
            magHistory[snapshot_index] = mag;
            energyHistory[snapshot_index] = energy;
        }
    }

    // elapsed time to perform the simulation
    clock_t end_time = clock();
    float sim_time = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\n CPU MONTECARLO RUNTIME \n %.4f s:\n", sim_time);

    // print out
    printf("\nSimulation Evolution:\n");
    for (int s = 0; s < n_save; s++) {
        printf("Step %d: M = %.4f, E = %.4f\n", (s + 1) * SAVE_INTERVAL,
               magHistory[s], energyHistory[s]);
    }


    printf("\n WRITING SIMULATION\n");
    SaveSimulationBinary_States("simulation_states_512.bin", stateHistory, n_save, SHAPE);
    SaveSimulationBinary_EnergyMag("simulation_magnetization_512.bin", magHistory, n_save);
    SaveSimulationBinary_EnergyMag("simulation_energy_512.bin", energyHistory, n_save);
    printf("\n complete !\n");
    
    free(state);
    free(stateHistory);
    free(magHistory);
    free(energyHistory);

    printf("\nSimulation completed!\n");
    return 0;
}
