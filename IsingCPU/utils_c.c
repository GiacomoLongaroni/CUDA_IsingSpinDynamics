#include "Ising_c.h"

/*
    FUNCTIONS USED IN THE MAIN PROGRAM AND BENCHMARKS
*/

// Ising state initialization: each site is initialized with +1 or -1
void InitializeRandomState(int *state, int dim) {
    int size = dim * dim;
    for (int i = 0; i < size; i++) {
        state[i] = (rand() % 2) * 2 - 1;
    }
}

// Function to compute the magnetization of an ising state 
// Σs_i / size
void ComputeMagnetization(int *state, float *magnetization, int dim) {
    float mag = 0.0f;
    int size = dim * dim;
    for (int i = 0; i < size; i++) {
        mag += state[i];
    }
    *magnetization = mag / (float)size;
}

// Function to compute the energy of an Ising state 
// -J*ΣΣ(s_i s_j) - H Σs_i / size (s_j nearest neighbor)
void computeStateEnergy(int *state, float *energy, int dim, float k, float h) {

    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {

            // lattice site index
            int idx = i * dim + j;
            // nearest neighbor index
            int up = (i == 0) ? ((dim - 1) * dim + j) : idx - dim;
            int right = (j == dim - 1) ? (i * dim) : idx + 1;
            
            sum -= 2 * k * state[idx] * (state[up] + state[right] + h / (2.0f * k));
        }
    }
    *energy = sum / (float)(dim * dim);
}

// Function to perform the update of the lattice using Metropolis algorithm
void PerformMonteCarloStep(int *state, int dim, float t, float k, float h) {
    int size = dim * dim;
    
    // n attempts of flip (n = size of lattice)
    for (int attempt = 0; attempt < size; attempt++) {

        // random site picking
        int i = rand() % dim;
        int j = rand() % dim;
        int idx = i * dim + j;
        
        int current_spin = state[idx];
        int new_spin = -current_spin;
        
        // nn index
        int left  = state[i * dim + ((j - 1 + dim) % dim)];
        int right = state[i * dim + ((j + 1) % dim)];
        int up    = state[((i - 1 + dim) % dim) * dim + j];
        int down  = state[((i + 1) % dim) * dim + j];
        
        // energy variation : ΔE = 2 * s_i * (s_left + s_right + s_up + s_down)
        int deltaE = 2 * current_spin * (left + right + up + down);
        
        // Metropolis criterium to accept the flip
        if (deltaE <= 0 || ((rand() / (float)RAND_MAX) < exp(-deltaE / t))) {
            state[idx] = new_spin;
        }
    }
}

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
