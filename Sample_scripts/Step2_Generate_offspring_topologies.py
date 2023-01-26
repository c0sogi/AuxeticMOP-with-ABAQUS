from auxeticmop.GeneticAlgorithm import generate_offspring
from auxeticmop.ParameterDefinitions import Parameters
from auxeticmop.PostProcessing import visualize_one_cube
from auxeticmop.FileIO import pickle_io

"""
Randomly generate parent topologies
"""

# Parameters
number_of_voxels_x = 10
number_of_voxels_y = 10
number_of_voxels_z = 10
topology_density = 0.4
number_of_topologies_to_create = 5
mutation_rate = 0.1


# Initialize parameters
parameters = Parameters(lx=number_of_voxels_x, ly=number_of_voxels_y, lz=number_of_voxels_z,
                        end_pop=number_of_topologies_to_create, mutation_rate=mutation_rate)
parameters.post_initialize()

# Load parent topologies from pickle file
parent_topologies = pickle_io('Topologies_1', mode='r')['parent']

# Create offspring topologies and visualize as full-sized cube
offspring_topologies = generate_offspring(gen=1, topo_parents=parent_topologies, mutation_rate=parameters.mutation_rate,
                                          lx=parameters.lx, ly=parameters.ly, lz=parameters.lz,
                                          end_pop=parameters.end_pop, save_file_as=None)
for offspring_topology in offspring_topologies:
    visualize_one_cube(cube_3d_array=offspring_topology, full=True)
