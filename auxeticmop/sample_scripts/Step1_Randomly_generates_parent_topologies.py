"""
Randomly generate parent topologies
"""


def run():
    from ..GeneticAlgorithm import random_parent_generation
    from ..ParameterDefinitions import Parameters
    from ..PostProcessing import visualize_one_cube
    import os
    script_dir = os.path.dirname(__file__)
    sample_data_dir = os.path.join(script_dir, 'sample_data')
    os.chdir(sample_data_dir)

    # Parameters
    number_of_voxels_x = 5
    number_of_voxels_y = 5
    number_of_voxels_z = 5
    topology_density = 0.3
    number_of_topologies_to_create = 10
    mutation_rate = 0.1
    save_parent_topologies_file_as = 'Topologies_1'
    show_created_parent_topologies = True
    full_sized_cube = True

    # Initialize parameters
    parameters = Parameters(lx=number_of_voxels_x, ly=number_of_voxels_y, lz=number_of_voxels_z,
                            end_pop=number_of_topologies_to_create, mutation_rate=mutation_rate)
    parameters.post_initialize()

    # Create parent topologies and visualize as full-sized cube
    parent_topologies = random_parent_generation(density=topology_density, params=parameters,
                                                 save_file_as=save_parent_topologies_file_as)
    if show_created_parent_topologies:
        for parent_topology in parent_topologies:
            visualize_one_cube(parent_topology, full=full_sized_cube)


if __name__ == '__main__':
    run()
