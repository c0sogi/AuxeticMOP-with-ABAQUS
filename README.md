# AuxeticMOP
Finding metamaterial structure with negative poisson's ratio using ABAQUS and MOP evolutionary algorithm approaches.
I'm currently working on it.

The script "main.py" generate 1/8 structure of an auxetic(meta material with negative poisson's ratio) unit cell using ABAQUS CAE software.
Genetic algorithm(NSGA) is used to validate and assess fitness values of generated topologies, thereby selecting best topologies.
The overall step is 
- 1. generating offspring topologies from parent topologies.
- 2. Analyze mechanical properties of offspring topologies using ABAQUS CAE.
- 3. Assess fitness values of parents and offsprings.
- 4. Select desired topologies which fits pareto-front(non-dominated) points and export these as next parent.
- 5. Redo 1~4 steps for next generations.

## Required
- Language: Python 3.10
- External libraries: numpy, numba, scipy, matplotlib
- Other software: ABAQUS CAE
