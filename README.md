# AuxeticMOP-with-ABAQUS
Finding metamaterial structure with negative poisson's ratio using ABAQUS and MOP evolutionary algorithm approaches.
I'm currently working on it.

The script "main.py" generates 1/8 structure of unit cell using ABAQUS CAE software by genetic algorithm.
This script is especially for finding mechanical meta-material structure consisting of 3D voxels.
Non-dominated Sorting Genetic Algorithm(NSGA) is used to validate and assess fitness values of generated topologies.
The validation steps consist of,
- 3D print-ability without generating supports.
- Allowing only Face-to-Face contact between voxels.
- All six faces of structure are connected as one tree, thereby not allowing force-free structure inside an unit cell.

I am currently researching on auxetic(negative poisson's ratio) structure with evaluation version 'ver3'.

The overall steps are, 
1. Generate offspring topologies from parent topologies.
2. Analyze mechanical properties of offspring topologies using ABAQUS CAE.
3. Assess fitness values of parents and offsprings.
4. Select desired topologies which fits pareto-front(non-dominated) points and export these as next parent.
5. Redo 1~4 steps for next generations.

## Required
- Language: Python 3.10
- External libraries: numpy, numba, scipy, matplotlib
- Other software: ABAQUS CAE
