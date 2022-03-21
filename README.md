# QKE-Quantum-Noise-Fingerprint-CSC-586A
This repository contains code to do quantum kernel estimation to classify quantum computers based on their noise fingerprint. It is part of coursework for the University of Victoria course CSC 586A.

## Usage
To simulate QKE, run *QKE_Simulation_Implementation.ipynb*. To run QKE on a quantum device, use *QKE_Runtime_Implementation.ipynb*. The implementation on a quantum device uses Qiskit's Quantum Runtime. The main program is contained in *quantum_kernel_estimation.py*.

A random sample of the total data is used to avoid running the program for an excessive amount of time. Change the number 'n' in the Batch cell to use different sample sizes. The implementation on a quantum device times out at 2 hours so a job for n > 80 will likely fail. 