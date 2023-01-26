def generate_offspring_test():
    from auxeticmop.GeneticAlgorithm import generate_offspring
    from auxeticmop.FileIO import pickle_aio
    from auxeticmop.PostProcessing import visualize_one_cube
    import asyncio
    import os

    # Loaded parent
    os.chdir('../abaqus data')
    topos = asyncio.run(pickle_aio('Topologies_1', mode='r'))
    topo_parent = topos['parent']
    print('[1] topo_parent shape: ', topo_parent.shape)

    # Generate offspring
    topo_offsprings = generate_offspring(topo_parents=topo_parent, gen=1, lx=10, ly=10, lz=10, end_pop=100,
                                         mutation_rate=0.1)
    print('[2] topo_offsprings shape: ', topo_offsprings.shape)
    for topo_offspring in topo_offsprings:
        visualize_one_cube(cube_3d_array=topo_offspring)


def flattened_topos_to_3d_topos():
    from auxeticmop.FileIO import pickle_aio, get_sorted_file_numbers_from_pattern
    import asyncio
    import os

    os.chdir('../abaqus data')
    sorted_topo_numbers = get_sorted_file_numbers_from_pattern(r'Topologies_\d+')
    topos_filenames = [f'Topologies_{num}' for num in sorted_topo_numbers]
    for topos_filename in topos_filenames:
        topos = asyncio.run(pickle_aio(topos_filename, mode='r'))
        new_dict = dict()
        try:
            topos_parent = topos['parent'].reshape((-1, 10, 10, 10))
            print('parent topo reshaped to: ', topos_parent.shape)
            new_dict.update({'parent': topos_parent})
        except KeyError:
            pass
        try:
            topos_offspring = topos['offspring'].reshape((-1, 10, 10, 10))
            print('offspring topo reshaped to: ', topos_offspring.shape)
            new_dict.update({'offspring': topos_offspring})
        except KeyError:
            pass
        asyncio.run(pickle_aio(topos_filename, mode='w', to_dump=new_dict))


def cutting_test():
    pass


if __name__ == '__main__':
    generate_offspring_test()
