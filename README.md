# CUDA Ising Spin Dynamics

<div align="center">
  <img src="Plots/ising_animation.gif" alt="Ising Model Simulation Animation" width="300"/>
</div>

CUDA Ising Spin Dynamics is my project developed for the _"Modern Computing for Physics"_ course in the Physics of Data MS. This repository provides two implementations of a 2D Ising model simulation:

- **CPU Version:** A baseline single-core implementation.
- **GPU Version:** An optimized version leveraging CUDA, specifically tested on the Nvidia Jetson Nano Developer Kit (2GB).

## Table of Contents

- [About](#about)
- [Requirements](#requirements)
- [Folder Structure](#folder-structure)
- [Build Script](#build-script)
- [Usage](#usage)

## About

The 2D Ising model is a classic model in statistical physics used to describe phase transitions in ferromagnetic materials. This project implements the Ising spin model simulation on a squared lattice to compare performance between a CPU-based solution and a GPU-accelerated implementation using CUDA.

## Requirements

### Hardware
- Nvidia GPU

### Software
- **CUDA Toolkit** (required for building and running the GPU version)
- **C/C++ Compilers:**
  - GCC (for the CPU version)
  - NVCC (for compiling CUDA code in the GPU version)

## Folder Structure

Below is an overview of the repository structure along with the purpose of the key files:

- **IsingCPU/**
  - `ising_cpu.c`  
    *Contains the C implementation of the 2D Ising model simulation for the CPU version.*
  - `Makefile`  
    *Build instructions to compile the CPU version. If absent, manual compilation using GCC is supported.*
  - *(Additional auxiliary source or header files can be included here if needed for CPU computations.)*

- **IsingGPU/**
  - `ising_gpu.cu`  
    *Contains the CUDA source code for the 2D Ising model simulation optimized for GPU execution.*
  - `Makefile`  
    *Build instructions to compile the GPU version with NVCC. If not provided, the bash script will use NVCC directly.*
  - *(Other CUDA-specific files, such as headers or utility sources, might be present for organizing GPU computations.)*

- **Plots/**
  - `ising_animation.gif`  
    *A GIF animation that visualizes the dynamic evolution of the spin simulation.*
  - `1000_bendwith.png`  
    *A sample plot image capturing simulation output data (e.g., energy, magnetization, etc.).*
  - `.DS_Store`  
    *An automatically generated macOS file; it can be ignored.*

- **README.md**  
  *This file, providing an overview, instructions, and file descriptions of the repository.*

- **build.sh**  
  *A bash script that automates the build process for both CPU and GPU implementations.*

## Build Script

Below is the bash build script to compile both the CPU and GPU versions. Save the script as `build.sh` in the repository's root and execute it from the terminal.

```bash
#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Building CPU version..."
if [ -d "IsingCPU" ]; then
    cd IsingCPU
    if [ -f "Makefile" ]; then
        make
    else
        echo "No Makefile found in IsingCPU. Attempting manual compilation..."
        gcc -o ising_cpu ising_cpu.c -O2 || { echo "Error building CPU version"; exit 1; }
    fi
    cd ..
else
    echo "IsingCPU directory not found!"
fi

echo "Building GPU version..."
if [ -d "IsingGPU" ]; then
    cd IsingGPU
    if [ -f "Makefile" ]; then
        make
    else
        echo "No Makefile found in IsingGPU. Attempting manual compilation..."
        nvcc -o ising_gpu ising_gpu.cu -O2 || { echo "Error building GPU version"; exit 1; }
    fi
    cd ..
else
    echo "IsingGPU directory not found!"
fi

echo "Build completed successfully."


