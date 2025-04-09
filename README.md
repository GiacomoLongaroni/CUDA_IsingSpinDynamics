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
  - `utils_c.c`*Contains the C functions for the implementation of the 2D Ising model simulation on CPU.*
  - `main_c.c`*Contains the C main on CPU.*

- **IsingGPU/**
  - `utils_cuda.cu` *Contains the C functions for the implementation of the 2D Ising model simulation running on the CPU.*
  - `kernels_cuda.cu`*Contains the CUDA Kernels running on the GPU*
  - `main_cuda.cu` *Contains the CUDA main.*




