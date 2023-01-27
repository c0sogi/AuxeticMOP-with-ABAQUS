from auxeticmop.GeneticAlgorithm import generate_offspring
from auxeticmop import Parameters
from auxeticmop import visualize_one_cube
from auxeticmop.FileIO import pickle_io

"""
Randomly generate parent topologies
"""

# Parameters
number_of_voxels_x = 10
number_of_voxels_y = 10
number_of_voxels_z = 10
number_of_topologies_to_create = 100
mutation_rate = 0.1
parent_topology_file_name = 'Topologies_1'
save_offspring_topologies_file_as = 'Topologies_1'
show_created_offspring_topologies = True


# Initialize parameters
parameters = Parameters(lx=number_of_voxels_x, ly=number_of_voxels_y, lz=number_of_voxels_z,
                        end_pop=number_of_topologies_to_create, mutation_rate=mutation_rate)
parameters.post_initialize()

# Load parent topologies from pickle file
parent_topologies = pickle_io(parent_topology_file_name, mode='r')['parent']

# Create offspring topologies and visualize as full-sized cube
offspring_topologies = generate_offspring(gen=1, topo_parents=parent_topologies, params=parameters,
                                          save_file_as=save_offspring_topologies_file_as)
if show_created_offspring_topologies:
    for offspring_topology in offspring_topologies:
        visualize_one_cube(cube_3d_array=offspring_topology, full=True)
