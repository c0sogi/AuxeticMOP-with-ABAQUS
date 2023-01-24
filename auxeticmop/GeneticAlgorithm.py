import random
import numpy as np
import itertools
import os
from functools import reduce
from .MutateAndValidate import mutate_and_validate_topology, visualize_one_cube
from .FileIO import dump_pickled_dict_data, load_pickled_dict_data
from .ParameterDefinitions import Parameters


def inspect_clone_in_all_parents(w, topology_flattened, all_parent_topologies):
    for generation_idx in range(w):
        parents_topologies = all_parent_topologies[generation_idx]
        for parent_idx in range(len(parents_topologies)):
            if np.array_equal(parents_topologies[parent_idx], topology_flattened):
                return True
    return False


def inspect_clone_in_current_offsprings(topology, offspring):  # inspect topology clones
    for offspring_idx in range(len(offspring)):
        if np.array_equal(offspring[offspring_idx], topology):
            return True
    return False


def cutting_function(topologies):
    _topologies = topologies.reshape((len(topologies), -1))
    cuttings = np.arange(0, _topologies.shape[1], step=1, dtype=int)
    np.random.shuffle(cuttings)
    cutting, candidate = None, None
    candidate_found_flag = False
    while not candidate_found_flag:
        candidate = list()
        cutting = cuttings[-1]
        cuttings = np.delete(cuttings, obj=-1, axis=0)
        for parent_idx in range(len(_topologies)):
            if _topologies[parent_idx, cutting] == 1:
                candidate.append(parent_idx)
            if len(candidate) == 2:
                candidate_found_flag = True
        if len(cuttings) == 0:
            print("[Cutting function] Can't make connection")
            exit()
    return cutting, candidate


def candidates(candidate_list):
    candidates_results = list(itertools.combinations(candidate_list, 2))  # possible candidate pair of parents
    random.shuffle(candidates_results)
    return candidates_results


def crossover(chromosome_1, chromosome_2, cutting_section):  # Crossover process
    chromosome_1_flattened = chromosome_1.flatten()
    chromosome_2_flattened = chromosome_2.flatten()
    cross_overed_chromosome_1 = np.empty_like(chromosome_1_flattened)
    cross_overed_chromosome_2 = np.empty_like(chromosome_2_flattened)
    cross_overed_chromosome_1[0:cutting_section] = chromosome_1_flattened[0:cutting_section]
    cross_overed_chromosome_1[cutting_section:] = chromosome_2_flattened[cutting_section:]
    cross_overed_chromosome_2[0:cutting_section] = chromosome_2_flattened[0:cutting_section]
    cross_overed_chromosome_2[cutting_section:] = chromosome_1_flattened[cutting_section:]
    return cross_overed_chromosome_1.reshape(chromosome_1.shape), cross_overed_chromosome_2.reshape(chromosome_2.shape)


def generate_offspring(topo_parent: np.ndarray, gen: int, lx: int, ly: int, lz: int,
                       end_pop: int, mutation_rate: float, timeout: float) -> np.ndarray:
    """
    Generating topo_offspring from parents. Crossover & Mutation & Validating processes will be held.
    Validating processes contain, checking 3d-print-ability without support, one voxel tree contacting six faces
    in a cube
    :param topo_parent: Topology array of parent, shape: (end_pop, lx, ly, lz)
    :param gen: Current generation
    :param lx: Total Voxels in x direction of a cube
    :param ly: Total Voxels in y direction of a cube
    :param lz: Total Voxels in z direction of a cube
    :param end_pop: The size of population in one generation
    :param mutation_rate: The mutation rate in range between 0~1, in the function "mutate_and_validate_topology"
    :param timeout: The time in seconds which will prevent validation process from infinite-looping.
    The more timeout, the longer time will be allowed for the function "mutate_and_validate_topology".
    :return:
    """
    topo_offspring = np.empty((0, lx, ly, lz), int)
    trial = 1
    validation_count = 0
    all_parents_topologies = np.empty((0, end_pop, lx, ly, lz), int)
    for generation_idx in range(gen):
        parent_topologies = load_pickled_dict_data(f'Topologies_{generation_idx + 1}')['parent']
        all_parents_topologies = np.vstack((all_parents_topologies, np.expand_dims(parent_topologies, axis=0)))
    while True:
        print('[Generate topo_offspring] Trial: ', trial)
        trial += 1
        cutting_section, candidate_list = cutting_function(topologies=topo_parent)
        print('[Generate topo_offspring] Candidate list: ', candidate_list)
        candidate_pairs_list = candidates(candidate_list=candidate_list)
        for pair_idx in range(len(candidate_pairs_list)):
            chromosome_1_idx = candidate_pairs_list[pair_idx][0]
            chromosome_2_idx = candidate_pairs_list[pair_idx][1]
            cross_overed_chromosome_1, cross_overed_chromosome_2 = crossover(chromosome_1=topo_parent[chromosome_1_idx],
                                                                             chromosome_2=topo_parent[chromosome_2_idx],
                                                                             cutting_section=cutting_section)
            validated_chromosome_1 = mutate_and_validate_topology(cross_overed_chromosome_1,
                                                                  mutation_probability=mutation_rate, timeout=timeout)
            validated_chromosome_2 = mutate_and_validate_topology(cross_overed_chromosome_2,
                                                                  mutation_probability=mutation_rate, timeout=timeout)
            for validated_chromosome in (validated_chromosome_1, validated_chromosome_2):
                if validated_chromosome is None:
                    print('[Generate topo_offspring] <!> Non-connected tree detected')
                    continue
                else:
                    is_any_clone_in_all_parents = inspect_clone_in_all_parents(
                        w=gen, topology_flattened=validated_chromosome, all_parent_topologies=all_parents_topologies)
                    is_any_clone_in_current_offsprings = inspect_clone_in_current_offsprings(
                        topology=validated_chromosome, offspring=topo_offspring)
                    if is_any_clone_in_all_parents:
                        print('[Generate topo_offspring] Clone structure found in parents!')
                        continue
                    elif is_any_clone_in_current_offsprings:
                        print('[Generate topo_offspring] Clone structure found in current offsprings!')
                        continue
                    else:
                        topo_offspring = np.vstack((topo_offspring, np.expand_dims(validated_chromosome, axis=0)))
                        validation_count += 1
                        print(f'[Generate topo_offspring] Validation of chromosome {validation_count} complete!')
                        if len(topo_offspring) == end_pop:
                            print('[Generate topo_offspring] Generating topo_offspring complete')
                            dump_pickled_dict_data(f'Topologies_{gen}', key='offspring',
                                                   to_dump=topo_offspring, mode='a')
                            return topo_offspring


def random_array(shape, probability):
    return np.random.choice([1, 0], size=reduce(lambda x, y: x * y, shape), p=[probability, 1 - probability]).reshape(
        shape)


def random_parent_generation(gen: int, density: float, params: Parameters, show_parent: bool = False) -> np.ndarray:
    parents = np.empty((params.end_pop, params.lx, params.ly, params.lz))
    total_parent_generation_count = 0
    total_volume_frac = 0
    while total_parent_generation_count < params.end_pop:
        rand_arr = random_array(shape=(params.lx, params.ly, params.lz), probability=density)
        parent = mutate_and_validate_topology(rand_arr, mutation_probability=params.mutation_rate,
                                              timeout=params.timeout)
        if parent is None:
            continue
        print(f'<<<<< Parent {total_parent_generation_count + 1} >>>>>')
        volume_frac = np.count_nonzero(parent) / (params.lx * params.ly * params.lz / 100)
        total_volume_frac += volume_frac
        print(f'Volume fraction: {volume_frac:.1f} %\n')
        parents[total_parent_generation_count] = parent
        total_parent_generation_count += 1
    print(f'Average volume fraction: {total_volume_frac / params.end_pop:.1f} %')
    dump_pickled_dict_data(file_name=f'Topologies_{gen}', key='parent', to_dump=parents, mode='w')
    if show_parent:
        for parent in parents:
            visualize_one_cube(parent, full=False)
    return parents


def inspect_topologies(generation):
    # offspring끼리 비교
    for w in range(1, generation):
        topo_1 = np.genfromtxt(path + f'topo_offspring_{w}.csv', delimiter=',', dtype=int)
        topo_2 = np.genfromtxt(path + f'topo_offspring_{w + 1}.csv', delimiter=',', dtype=int)
        count = 0
        for i in range(100):
            flag = False
            for j in range(100):
                if np.array_equal(topo_1[i], topo_2[j]):
                    flag = True
            if flag:
                count += 1
        print(f'offspring {w} vs {w + 1}: ', count)

    # parent끼리 비교
    for w in range(1, generation):
        topo_1 = np.genfromtxt(path + f'topo_parent_{w}.csv', delimiter=',', dtype=int)
        topo_2 = np.genfromtxt(path + f'topo_parent_{w + 1}.csv', delimiter=',', dtype=int)
        count = 0
        for i in range(100):
            flag = False
            for j in range(100):
                if np.array_equal(topo_1[i], topo_2[j]):
                    flag = True
            if flag:
                count += 1
        print(f'parent {w} vs {w + 1}: ', count)

    # Parent와 offspring끼리 비교
    for w in range(1, generation+1):
        topo_1 = np.genfromtxt(path + f'topo_offspring_{w}.csv', delimiter=',', dtype=int)
        topo_2 = np.genfromtxt(path + f'topo_parent_{w}.csv', delimiter=',', dtype=int)
        count = 0
        for i in range(100):
            flag = False
            for j in range(100):
                if np.array_equal(topo_1[i], topo_2[j]):
                    flag = True
            if flag:
                count += 1
        print(f'offspring {w} vs parent {w}: ', count)

    # offspring 내부 비교
    for w in range(1, generation+1):
        topo_1 = np.genfromtxt(path + f'topo_offspring_{w}.csv', delimiter=',', dtype=int)
        count = 0
        for i in range(100):
            flag = False
            for j in range(100):
                if i != j and np.array_equal(topo_1[i], topo_1[j]):
                    flag = True
            if flag:
                count += 1
        print(f'offspring {w}: ', count)

    # parent 내부 비교
    for w in range(1, generation+1):
        topo_1 = np.genfromtxt(path + f'topo_parent_{w}.csv', delimiter=',', dtype=int)
        count = 0
        for i in range(100):
            flag = False
            for j in range(100):
                if i != j and np.array_equal(topo_1[i], topo_1[j]):
                    flag = True
            if flag:
                count += 1
        print(f'parent {w}: ', count)

    # offspring vs parent 다음세대 비교
    for w in range(1, generation):
        topo_1 = np.genfromtxt(path + f'topo_offspring_{w}.csv', delimiter=',', dtype=int)
        topo_2 = np.genfromtxt(path + f'topo_parent_{w + 1}.csv', delimiter=',', dtype=int)
        count = 0
        for i in range(100):
            flag = False
            for j in range(100):
                if np.array_equal(topo_1[i], topo_2[j]):
                    flag = True
            if flag:
                count += 1
        print(f'offspring {w} vs parent {w + 1}: ', count)


if __name__ == '__main__':
    from FileIO import parent_import
    path = 'F:/shshsh/data-23-1-4/'
    os.chdir(path)
    topos, _ = parent_import(gen_num=18)  # topo: (100, 1000), result: (100, 12)
    number_of_voxels_x = 10
    number_of_voxels_y = 10
    number_of_voxels_z = 10
    end_population = 100
    offsprings = generate_offspring(topo_parent=topos, gen=18, end_pop=end_population,
                                    mutation_rate=0.05, timeout=0.5,
                                    lx=number_of_voxels_x, ly=number_of_voxels_y, lz=number_of_voxels_z)
    inspect_topologies(generation=19)
    for off_idx in range(100):
        visualize_one_cube(offsprings[off_idx])
