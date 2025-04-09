#include "Ising_cuda.h"

/*

    FUNCTIONS RUNNING ON CPU USED IN THE MAIN PROGRAM AND BENCHMARKS

*/


// Function to save the simulation (lattice state)
void SaveSimulationBinary_States(const char *filename, const int *stateHistory, int n_save, int shape) {
    FILE *fp = fopen(filename, "wb");
    fwrite(stateHistory, sizeof(int), n_save * shape, fp);
    fclose(fp);
}

// Function to save the simulation (energy and magnetization)
void SaveSimulationBinary_EnergyMag(const char *filename, const float *stateHistory, int n_save) {
    FILE *fp = fopen(filename, "wb");
    fwrite(stateHistory, sizeof(float), n_save, fp);
    fclose(fp);
}

// Lattice initialization each site can be +1 or -1
void InitializeRandomState(int *state, int shape) {
    for (int i = 0; i < shape; i++) {
        state[i] = (rand() % 2) * 2 - 1;
    }
}

// debug print out 
void PrintState(int *state, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%2d ", state[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// function to initialize a squared matrix of seeds
// will be used in the montecarlo kernel to compute the metropolis criterium  
void InitializeSeed(int *InitialSeed, int shape) {
    for (int i = 0; i < shape; i++) {
        InitialSeed[i] = rand();
    }
}

// function to compute the final magnetization 
// it sums the reduced magnetization of the different blocks
void computeMagnetization(float *blockHistory, float *History, int n_blocks, int n_save) {
    for (int s = 0; s < n_save; s++){
        float mag = 0.;
        for (int b = 0; b < n_blocks; b++){
            mag += blockHistory[s * n_blocks + b];
        }
        History[s] = mag;
    } 
}

// function to compute the final magnetization 
// it sums the reduced energy of the different blocks
void computeEnergy(float *blockHistory, float *History, int n_blocks, int n_save) {
    for (int s = 0; s < n_save; s++){
        float energy = 0.;
        for (int b = 0; b < n_blocks; b++){
            energy += blockHistory[s * n_blocks + b];
        }
        History[s] = energy;
    }
}

// function to compute the energy given an ising state 
void computeStateEnergyCPU(int *state, float *energyCPU, int width, float k, float h) {
    float sum = 0.0f;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            int up = (i == 0) ? ((width - 1) * width + j) : idx - width;
            int right = (j == width - 1) ? (i * width) : idx + 1;
            sum -= k * state[idx] * (state[up] + state[right] + h / (2.0f * k));
        }
    }
    *energyCPU = sum / (width * width);
}