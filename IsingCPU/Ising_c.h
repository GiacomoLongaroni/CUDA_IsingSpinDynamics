#ifndef ISING_H
#define ISING_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

// Macros and costants for the simulation
//#define WIDTH 512
//#define SHAPE (WIDTH * WIDTH)
//#define N_STEPS 10000
//#define SAVE_INTERVAL 100
//
//#define J 1.0f
//#define H 0.3f

// Functions declaration implemented in utils_c.c
void SaveSimulationBinary_States(const char *filename, const int *stateHistory, int n_save, int shape);
void SaveSimulationBinary_EnergyMag(const char *filename, const float *stateHistory, int n_save);
void InitializeRandomState(int *state, int dim);
void ComputeMagnetization(int *state, float *magnetization, int dim);
void computeStateEnergy(int *state, float *energy, int dim, float k, float h);
void PerformMonteCarloStep(int *state, int dim, float t, float k, float h);

#endif // ISING_H