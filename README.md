# AuxeticMOP-with-ABAQUS `1.0.2`

## Purpose
- Finding metamaterial structure with negative poisson's ratio using ABAQUS and MOP evolutionary algorithm approaches.
- In addition to structure with negative poisson's ratio, other types of material structure can be created by varying version fitness values definitions.
- The definition of fitness value for negative Poisson's ratio is well defined in `auxeticmop.ParameterDefinitions.fitness_definitions['ver3']`.


## Features
- The script `full_scripts.py` or `auxeticmop.sample_scripts.full_steps.run()` generates 1/8 structure of unit cell using ABAQUS CAE software by genetic algorithm.
This script is especially for finding mechanical meta-material structure consisting of 3D voxels.
- GUI is provided for getting initial parameters for ABAQUS, and plotting results when a generation work is done.
  + Related contents: `auxeticmop.GraphicUserInterface`

- Python script running on ABAQUS is located in `auxeticmop.AbaqusScripts`. This will run only on ABAQUS-embedded python
interpreter, and maybe the version is `2.7.15`. Other scripts are running on newer Python.

- Non-dominated Sorting Genetic Algorithm(NSGA) is used to validate and assess fitness values of generated topologies.
  + Related contents: `auxeticmop.GeneticAlgorithm`, `auxeticmop.MutateAndValidate`

## Example


- ### Auxetic cell
<p align="center">
  <img src="https://user-images.githubusercontent.com/121936784/215425185-a7d62330-8bec-4650-9079-637cc1dd8575.png"/>
</p>

>- 10 by 10 by 10 voxels per 1/8 cell.
>- Negative negative poisson's ratio structure

- ### GUI example

<p align="center">
  <img src="https://user-images.githubusercontent.com/121936784/215425207-44187735-cea7-4486-a2b9-bf4683797a53.png"/>
</p>

>- Building a GUI using `tkinter` and `matplotlib`
>- Receiving parameter values required for ABAQUS analysis and GA setting from the user
>- The Pareto optimization solution and hyper volume value calculated from the Main Process
are input in real time and output as a graph.

## Install
Before installing this package, you must [ABAQUS CAE](https://edu.3ds.com/en/software/abaqus-learning-edition) must be installed.

 To install the current release via `PyPI` with Python version `>=3.6` and `<3.11`:
```shell
$ pip install auxeticmop
```
... or to install the current release via `anaconda` with Python version `>=3.6` and `<3.11`:

```shell
$ conda install -c cosogi auxeticmop
```

#### *Try out whole GA steps*

```shell
$ python
```

```python
>>> from auxeticmop.sample_scripts import full_steps
>>> if __name__ == '__main__':
  ...   full_steps.run()
```
#### *Modify your parameter definitions*
```python
>>> from auxeticmop import ParameterDefinitions
>>> dir(ParameterDefinitions)
Output: ['FitnessDefinitions', 'GuiParameters', 'JsonFormat', 'Parameters', 'Union', '__builtins__',
         '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__',
         'dataclass', 'exported_field_outputs_format', 'fitness_definitions', 'material_property_definitions',
         'np', 'radiobutton_name_dict', 'translate_dictionary']
```
- Go to `auxeticmop.ParameterDefinitions` and use editor to directly customize parameters.
- If using VS code, press `F12`, if using Pycharm, press `Ctrl+B` to go to file.

## Overall Steps of GA
> All Steps are included in `auxeticmop.GeneticAlgorithm.NSGAModel.evolve_a_generation()`.
>1. Generate offspring topologies from parent topologies.
>   - Related contents: `auxeticmop.GeneticAlgorithm.NSGAModel.generate_offspring_topologies()`
>2. Analyze displacements, reaction forces, or other mechanical properties of offspring topologies using ABAQUS CAE.
>   - Related contents: `auxeticmop.Network.start_abaqus_cae()`, `auxeticmop.Network.request_abaqus()`, `auxeticmop.AbaqusScripts`
>3. Evaluate fitness values of parents and offsprings.
>   - Related contents: `auxeticmop.PostProcessing.evaluate_all_fitness_values()`
>4. Select desired topologies which fits pareto-front(non-dominated) points and export these as next parent.
>   - Related contents: `auxeticmop.PostProcessing.selection()`
>5. Redo steps 1~4 for next generations. Iterations of all generations are done in `auxeticmop.GeneticAlgorithm.NSGAModel.evolve()`.

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

| Evaluation<br/>version | Fitness<br/> value 1                                 | Fitness<br/> value 2                  |
|------------------------|------------------------------------------------------|---------------------------------------|
| ver1                   | RF<sub>22</sub>/RF<sub>22,max</sub> + `k`*`vol_frac` | ν <sub>21</sub> + `k` * `vol_frac`    |
| ver2                   | `vol_frac`                                           | RF<sub>22</sub>/RF<sub>22,max</sub>   |
| ver3                   | ν <sub>21</sub> + `k` * `vol_frac`                   | ν <sub>23</sub> +`k` * `vol_frac`     |
| ver4                   | (σ<sub>mises</sub>)<sub>max</sub>                    | `vol_frac`                            |
| ver5                   | (σ<sub>mises</sub>)<sub>max</sub>                    | max(ν <sub>21</sub>, ν <sub>23</sub>) |
>- `vol_frac`: Volume fraction in cell (0~1)
>- `k`: penalty coefficient
>- `k`: penalty coefficient

---
## Required
- [x] **[Language]** Python, with version `>=3.6 and <3.11`.
- Version dependency
  - `numba` for Python `3.11` is not supported yet.
  - `dataclass` is not supported under Python `3.6`
- [x] **[External libraries]** `numpy`, `numba`, `scipy`, `matplotlib`, `aiofiles`, `dataclasses`
- [x] **[Other software]** `ABAQUS CAE`
