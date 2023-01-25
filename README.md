# AuxeticMOP-with-ABAQUS
## Purpose
- Finding metamaterial structure with negative poisson's ratio using ABAQUS and MOP evolutionary algorithm approaches.
- I am currently researching on auxetic(negative poisson's ratio) structure with evaluation version `ver3`.


## Features
- The script `main.py` generates 1/8 structure of unit cell using ABAQUS CAE software by genetic algorithm.
This script is especially for finding mechanical meta-material structure consisting of 3D voxels.
- GUI is provided for getting initial parameters for ABAQUS, and plotting results when a generation work is done.
  + Related contents: `auxeticmop.GraphicUserInterface`

- Python script running on ABAQUS is located in `./abaqus data/ABQ.py`.
Maybe the version of Python interpreter embedded in ABAQUS is `2.7.15`. Other scripts are running on newer Python.

- Non-dominated Sorting Genetic Algorithm(NSGA) is used to validate and assess fitness values of generated topologies.
  + Related contents: `auxeticmop.GeneticAlgorithm`, `auxeticmop.MutateAndValidate`

## Overall Steps of GA
>1. Generate offspring topologies from parent topologies.
>   - Related contents: `auxeticmop.GeneticAlgorithm.generate_offspring`
>2. Analyze displacements, reaction forces, or other mechanical properties of offspring topologies using ABAQUS CAE.
>   - Related contents: `main.start_abaqus_cae`, `main.wait_for_abaqus`, `./abaqus data/ABQ.py`
>3. Evaluate fitness values of parents and offsprings.
>   - Related contents: `auxeticmop.PostProcessing.evaluate_all_fitness_values`
>4. Select desired topologies which fits pareto-front(non-dominated) points and export these as next parent.
>   - Related contents: `auxeticmop.PostProcessing.selection`
>5. Redo i~iv steps for next generations.

## Conditions to Meet in Validation Steps
- 3D print-ability without supports, maximum overhang distance is also considered.
  + Related contents: `auxeticmop.MutateAndValidate.make_3d_print_without_support`
- Allowing only Face-to-Face contact between voxels.
  + Related contents: `auxeticmop.MutateAndValidate.make_voxels_surface_contact`
- All six faces of structure are connected as one tree, thereby not allowing force-free structure inside an unit cell.
  + Related contents: `auxeticmop.MutateAndValidate.one_connected_tree`
## Fitness Value Definitions
- Those two fitness values(objective functions) should go lower.
- The fitness value definitions are well organized in `auxeticmop.ParameterDefinitions.fitness_definitions`.
- You can choose the version of fitness value evaluation in GUI.

| Evaluation<br/>version | Fitness<br/> value 1                                 | Fitness<br/> value 2                |
|------------------------|------------------------------------------------------|-------------------------------------|
| ver1                   | RF<sub>22</sub>/RF<sub>22,max</sub> + `k`*`vol_frac` | ν <sub>21</sub> + `k` * `vol_frac`  |
| ver2                   | `vol_frac`                                           | RF<sub>22</sub>/RF<sub>22,max</sub> |
| ver3                   | ν <sub>21</sub> + `k` * `vol_frac`                   | ν <sub>23</sub> +`k` * `vol_frac`   |
| ver4                   | (σ<sub>mises</sub>)<sub>max</sub>                    | `vol_frac`                          |
>- `vol_frac`: Volume fraction in cell (0~1)
>- `k`: penalty coefficient

---
## Required
- [x] **[Language]** Python, with version `>=3.9` for some new PEP syntax. `3.10` is recommended.
- [x] **[External libraries]** `numpy`, `numba`, `scipy`, `matplotlib`, `aiofiles`
- [x] **[Other software]** `ABAQUS CAE`
