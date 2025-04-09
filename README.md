# CUDA Ising Spin Dynamics

[![Ising Model Simulation Animation](Plots/ising_animation.gif)](Plots/ising_animation.gif)

CUDA Ising Spin Dynamics is my project developed for the _"Modern Computing for Physics"_ course in the Physics of Data MS. This repository provides two implementations of a 2D Ising model simulation:

- **CPU Version:** A baseline single core implementation.
- **GPU Version:** An optimized version leveraging CUDA, specifically tested on the Nvidia Jetson Nano Developer Kit (2GB).

## Table of Contents

- [About](#about)
- [Requirements](#requirements)


## About

The 2D Ising model is a classic model in statistical physics used to describe phase transition in ferromagnetic materials. This project implements the Ising spin model simulation on two different platforms to compare performance and demonstrate the benefits of GPU acceleration using CUDA. 

## Requirements

### Hardware
- Nvidia GPU 

### Software
- **CUDA Toolkit** (required for building and running the GPU version)
- **C/C++ Compiler:**
  - GCC (for the CPU version)
  - NVCC (for compiling CUDA code in the GPU version)

