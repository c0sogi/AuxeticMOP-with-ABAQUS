from auxeticmop.GeneticAlgorithm import random_parent_generation
from auxeticmop.ParameterDefinitions import Parameters
from auxeticmop.PostProcessing import visualize_one_cube

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

# Create parent topologies and visualize as full-sized cube
parent_topologies = random_parent_generation(density=topology_density, params=parameters, save_file_as=None)
for parent_topology in parent_topologies:
    visualize_one_cube(cube_3d_array=parent_topology, full=True)
