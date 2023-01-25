# AuxeticMOP-with-ABAQUS
## Purpose
- Finding metamaterial structure with negative poisson's ratio using ABAQUS and MOP evolutionary algorithm approaches.
I am currently researching on auxetic(negative poisson's ratio) structure with evaluation version 'ver3'.

- The script "main.py" generates 1/8 structure of unit cell using ABAQUS CAE software by genetic algorithm.
This script is especially for finding mechanical meta-material structure consisting of 3D voxels.
GUI is provided for getting initial parameters for ABAQUS, and plotting results when a generation work is done.

- Non-dominated Sorting Genetic Algorithm(NSGA) is used to validate and assess fitness values of generated topologies.

## Conditions to meet in Validation step
- 3D print-ability without generating supports.
- Allowing only Face-to-Face contact between voxels.
- All six faces of structure are connected as one tree, thereby not allowing force-free structure inside an unit cell.

## Fitness value definitions
- Those two fitness values(objective functions) should go lower.
- The fitness value definitions are well organized in "auxeticmop.ParameterDefinitions".
- You can choose the version of fitness value evaluation in GUI.

| Evaluation<br/>version | Fitness<br/> value 1                                 | Fitness<br/> value 2                |
|------------------------|------------------------------------------------------|-------------------------------------|
| ver1                   | RF<sub>22</sub>/RF<sub>22,max</sub> + `k`*`vol_frac` | ν <sub>21</sub> + `k` * `vol_frac`  |
| ver2                   | `vol_frac`                                           | RF<sub>22</sub>/RF<sub>22,max</sub> |
| ver3                   | ν <sub>21</sub> + `k` * `vol_frac`                   | ν <sub>23</sub> +`k` * `vol_frac`   |
| ver4                   | σ<sub>mises,max</sub>                                | `vol_frac`                          |
- `vol_frac`: Volume fraction in cell (0~1)
- `k`: penalty coefficient

## Overall steps of GA
1. Generate offspring topologies from parent topologies.
2. Analyze mechanical properties of offspring topologies using ABAQUS CAE.
3. Assess fitness values of parents and offsprings.
4. Select desired topologies which fits pareto-front(non-dominated) points and export these as next parent.
5. Redo 1~4 steps for next generations.

>## Required
> - `Language` Python, with version >=3.9 for some new PEP syntax. 3.10 is recommended.
> - `External libraries` numpy, numba, scipy, matplotlib, aiofiles
> - `Other software` ABAQUS CAE
