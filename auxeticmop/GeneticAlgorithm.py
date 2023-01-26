import random
import numpy as np
import itertools
from functools import reduce
from .MutateAndValidate import mutate_and_validate_topology
from .FileIO import pickle_io, pickles_io
from .ParameterDefinitions import Parameters


def find_where_same_array_locates(arr_to_find, big_arr):
    small_dimensions = len(arr_to_find.shape)
    big_dimensions = len(big_arr.shape)
    axis_range = tuple(i for i in range(big_dimensions - small_dimensions, big_dimensions))
    return np.argwhere(np.all(big_arr == arr_to_find, axis=axis_range))


def cutting_function(topologies):
    topologies_flattened = topologies.reshape((len(topologies), -1))
    possible_cutting_sections = np.random.permutation(topologies_flattened.shape[1])
    for possible_cutting_section in possible_cutting_sections:
        candidates_for_cutting = np.empty((0,), dtype=int)
        for topo_idx in range(len(topologies)):
            if topologies_flattened[topo_idx, possible_cutting_section] == 1:
                candidates_for_cutting = np.hstack((candidates_for_cutting, topo_idx))
        if len(candidates_for_cutting) >= 2:
            return possible_cutting_section, candidates_for_cutting
    raise LookupError


def cutting_function2(topologies):
    topologies_flattened = topologies.reshape((len(topologies), -1))
    cuttings = np.arange(0, topologies_flattened.shape[1], step=1, dtype=int)
    np.random.shuffle(cuttings)
    cutting, candidate = int, list
    candidate_found_flag = False
    while not candidate_found_flag:
        candidate = list()
        cutting = cuttings[-1]
        cuttings = np.delete(cuttings, obj=-1, axis=0)
        for parent_idx in range(len(topologies_flattened)):
            if topologies_flattened[parent_idx, cutting] == 1:
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


def generate_offspring(gen: int, topo_parents: np.ndarray, lx: int, ly: int, lz: int, end_pop: int, mutation_rate: float,
                       save_file_as: str = None) -> np.ndarray:
    """
    Generating topo_offspring from parents. Crossover & Mutation & Validating processes will be held.
    Validating processes contain, checking 3d-print-ability without support, one voxel tree contacting six faces
    in a cube
    :param save_file_as: Save pickle file of offspring topologies as this.
    :param topo_parents: Topology array of parent, shape: (end_pop, lx, ly, lz)
    :param gen: Current generation
    :param lx: Total Voxels in x direction of a cube
    :param ly: Total Voxels in y direction of a cube
    :param lz: Total Voxels in z direction of a cube
    :param end_pop: The size of population in one generation
    :param mutation_rate: The mutation rate in range between 0~1, in the function "mutate_and_validate_topology"
    The more timeout, the longer time will be allowed for the function "mutate_and_validate_topology".
    :return:
    """
    topo_offspring = np.empty((0, lx, ly, lz), int)
    trial = 1
    validation_count = 0
    all_topos = pickles_io(file_names=[f'Topologies_{g}' for g in range(1, gen + 1)], mode='r', key_option='int')
    all_topos_parent = np.array([topos['parent'] for topos in all_topos.values()], dtype=int)
    while True:
        print('[Generate topo_offspring] Trial: ', trial)
        trial += 1
        cutting_section, candidate_list = cutting_function(topologies=topo_parents)
        print('[Generate topo_offspring] Candidate list: ', candidate_list)
        candidate_pairs_list = candidates(candidate_list=candidate_list)
        for pair_idx in range(len(candidate_pairs_list)):
            chromosome_1_idx = candidate_pairs_list[pair_idx][0]
            chromosome_2_idx = candidate_pairs_list[pair_idx][1]
            cross_overed_chromosome_1, cross_overed_chromosome_2 = crossover(chromosome_1=topo_parents[chromosome_1_idx],
                                                                             chromosome_2=topo_parents[chromosome_2_idx],
                                                                             cutting_section=cutting_section)
            validated_chromosome_1 = mutate_and_validate_topology(cross_overed_chromosome_1,
                                                                  mutation_probability=mutation_rate)
            validated_chromosome_2 = mutate_and_validate_topology(cross_overed_chromosome_2,
                                                                  mutation_probability=mutation_rate)
            for validated_chromosome in (validated_chromosome_1, validated_chromosome_2):
                if validated_chromosome is None:
                    print('[Generate topo_offspring] <!> Non-connected tree detected')
                    continue
                else:
                    if len(find_where_same_array_locates(arr_to_find=validated_chromosome,
                                                         big_arr=all_topos_parent)):
                        print('[Generate topo_offspring] Clone structure found in parents!')
                        continue
                    elif len(find_where_same_array_locates(arr_to_find=validated_chromosome,
                                                           big_arr=topo_offspring)):
                        print('[Generate topo_offspring] Clone structure found in current offsprings!')
                        continue
                    else:
                        topo_offspring = np.vstack((topo_offspring, np.expand_dims(validated_chromosome, axis=0)))
                        validation_count += 1
                        print(f'[Generate topo_offspring] Validation of chromosome {validation_count} complete!')
                        if len(topo_offspring) == end_pop:
                            print('[Generate topo_offspring] Generating topo_offspring complete')
                            if save_file_as is not None:
                                pickle_io(save_file_as, mode='a', to_dump={'offspring': topo_offspring})
                            return topo_offspring


def random_array(shape, probability):
    return np.random.choice([1, 0], size=reduce(lambda x, y: x * y, shape), p=[probability, 1 - probability]).reshape(
        shape)


def random_parent_generation(density: float, params: Parameters, save_file_as: str = None) -> np.ndarray:
    parents = np.empty((params.end_pop, params.lx, params.ly, params.lz))
    total_parent_generation_count = 0
    total_volume_frac = 0
    while total_parent_generation_count < params.end_pop:
        print(f'<<<<< Parent {total_parent_generation_count + 1} >>>>>')
        while True:
            rand_arr = random_array(shape=(params.lx, params.ly, params.lz), probability=density)
            parent = mutate_and_validate_topology(rand_arr, mutation_probability=params.mutation_rate)
            if parent is not None:
                break
        volume_frac = np.count_nonzero(parent) / (params.lx * params.ly * params.lz / 100)
        total_volume_frac += volume_frac
        print(f'Volume fraction: {volume_frac:.1f} %\n')
        parents[total_parent_generation_count] = parent
        total_parent_generation_count += 1
    print(f'Average volume fraction: {total_volume_frac / params.end_pop:.1f} %')
    if save_file_as is not None:
        pickle_io(save_file_as, mode='w', to_dump={'parent': parents})
    return parents
