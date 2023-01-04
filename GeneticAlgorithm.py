import random
import numpy as np
import itertools
import os
from functools import reduce
from MutateAndValidate import mutate_and_validate_topology, visualize_one_cube
from numba import njit


def parent_import(w, restart_pop):
    topo = np.genfromtxt('topo_parent_' + str(w) + '.csv', delimiter=',', dtype=int)
    reslt = np.genfromtxt('Output_parent_' + str(w) + '.csv', delimiter=',', dtype=np.float32)

    if restart_pop == 0:
        return topo, reslt

    else:  # restart
        offspring = np.genfromtxt('topo_offspring_' + str(w) + '.csv', delimiter=',', dtype=int)
        return topo, reslt, offspring


def array_to_csv(path, arr, dtype, mode, save_as_int=False):
    if mode == 'a' and os.path.isfile(path):
        previous_arr = np.genfromtxt(path, delimiter=',', dtype=dtype)
        # print('[array_to_csv] append shape: ', previous_arr.shape, arr.shape)
        arr = np.vstack((previous_arr, arr))
    fmt = '%i' if save_as_int else '%.18e'
    np.savetxt(path, arr, delimiter=',', fmt=fmt)


def inspect_clone_in_previous_generations(topology, end_pop, w, lx, ly, lz):  # inspect topology clones
    if w == 1:
        return False
    else:
        for generation_idx in range(1, w):
            parents_topologies = np.genfromtxt('topo_parent_' + str(generation_idx) + '.csv', delimiter=',',
                                               dtype=int).reshape((end_pop, lx * ly * lz))
            for parent_idx in range(len(parents_topologies)):
                if np.array_equal(parents_topologies[parent_idx], topology[0]) or np.array_equal(
                        parents_topologies[parent_idx], topology[1]):
                    return True
        return False


@njit
def inspect_clone_in_current_offsprings(topology, offspring):  # inspect topology clones
    for offspring_idx in range(len(offspring)):
        if np.array_equal(offspring[offspring_idx], topology) or np.array_equal(offspring[offspring_idx], topology[1]):
            return True
    return False


def cutting_function(topologies):
    cuttings = np.arange(0, topologies.shape[1], step=1, dtype=int)
    np.random.shuffle(cuttings)
    cutting, candidate = None, None
    candidate_found_flag = False
    while not candidate_found_flag:
        candidate = list()
        cutting = cuttings[-1]
        cuttings = np.delete(cuttings, obj=-1, axis=0)
        for parent_idx in range(len(topologies)):  # correction: topo > topologys
            if topologies[parent_idx, cutting] == 1:  # correction: topo > topologys
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


def crossover(chromosome_1, chromosome_2, cutting_section):
    offspring1 = np.zeros_like(chromosome_1)
    offspring2 = np.zeros_like(chromosome_2)
    offspring1[0:cutting_section] = chromosome_1[0:cutting_section]
    offspring1[cutting_section:] = chromosome_2[cutting_section:]
    offspring2[0:cutting_section] = chromosome_2[0:cutting_section]
    offspring2[cutting_section:] = chromosome_1[cutting_section:]
    return np.vstack([offspring1, offspring2])


def generate_offspring(topologies, w, lx, ly, lz, end_pop, mutation_rate, add_probability, timeout):
    offspring = np.empty((0, lx * ly * lz), int)
    offspring_generation_complete = False
    trial = 1
    validation_count = 0
    while not offspring_generation_complete:
        print('[Generate offspring] Trial: ', trial)
        trial += 1
        cutting_section, candidate_list = cutting_function(topologies=topologies)
        print('[Generate offspring] Candidate list: ', candidate_list)
        candidate_pairs_list = candidates(candidate_list=candidate_list)
        for pair_idx in range(len(candidate_pairs_list)):
            chromosome_1_idx = candidate_pairs_list[pair_idx][0]
            chromosome_2_idx = candidate_pairs_list[pair_idx][1]
            cross_overed_pairs = crossover(chromosome_1=topologies[chromosome_1_idx],
                                           chromosome_2=topologies[chromosome_2_idx],
                                           cutting_section=cutting_section)
            cross_overed_chromosome_1 = cross_overed_pairs[0].reshape((lx, ly, lz))
            cross_overed_chromosome_2 = cross_overed_pairs[1].reshape((lx, ly, lz))
            validated_chromosome_1 = mutate_and_validate_topology(cross_overed_chromosome_1,
                                                                  mutation_probability=mutation_rate,
                                                                  add_probability=add_probability, timeout=timeout)
            validation_count += 1
            print(f'[Generate offspring] Validation of chromosome {validation_count} complete!')
            validated_chromosome_2 = mutate_and_validate_topology(cross_overed_chromosome_2,
                                                                  mutation_probability=mutation_rate,
                                                                  add_probability=add_probability, timeout=timeout)
            validation_count += 1
            print(f'[Generate offspring] Validation of chromosome {validation_count} complete!')
            validated_chromosomes = np.vstack((validated_chromosome_1.flatten(),
                                               validated_chromosome_2.flatten()))
            is_any_clone_in_previous_generations = inspect_clone_in_previous_generations(
                topology=validated_chromosomes, end_pop=end_pop, w=w, lx=lx, ly=ly, lz=lz)
            if is_any_clone_in_previous_generations:
                print('[Generate offspring] Clone structure found in previous generations!')
                continue
            is_any_clone_in_current_offsprings = inspect_clone_in_current_offsprings(
                topology=validated_chromosomes, offspring=offspring)
            if is_any_clone_in_previous_generations or is_any_clone_in_current_offsprings:
                print('[Generate offspring] Clone structure found in current offsprings!')
                continue
            offspring = np.vstack((offspring, validated_chromosome_1.flatten()))
            if len(offspring) == end_pop:
                offspring_generation_complete = True
                print('[Generate offspring] Generating offspring complete')
                break
            offspring = np.vstack((offspring, validated_chromosome_2.flatten()))
            if len(offspring) == end_pop:
                offspring_generation_complete = True
                print('[Generate offspring] Generating offspring complete')
                break
    array_to_csv(f'topo_offspring_{w}.csv', offspring, dtype=int, mode='w', save_as_int=True)
    return offspring.reshape((end_pop, lx, ly, lz))


def random_array(shape, probability):
    return np.random.choice([1, 0], size=reduce(lambda x, y: x * y, shape), p=[probability, 1 - probability]).reshape(
        shape)


def random_parents_generation(lx, ly, lz, total_parents, density, mutation_probability, add_probability, timeout,
                              save_file=True):
    parent_name = 'topo_parent_1.csv'
    parents = np.empty((total_parents, lx * ly * lz))
    total_volume_frac = 0
    for parent_idx in range(total_parents):
        rand_arr = random_array(shape=(lx, ly, lz), probability=density)
        print(f'<<<<< Parent {parent_idx + 1} >>>>>')
        parent = mutate_and_validate_topology(rand_arr, mutation_probability=mutation_probability,
                                              add_probability=add_probability, timeout=timeout)
        volume_frac = np.count_nonzero(parent) / (lx * ly * lz / 100)
        total_volume_frac += volume_frac
        print(f'Volume fraction: {volume_frac:.1f} %\n')
        parents[parent_idx] = parent.flatten()
    print(f'Average volume fraction: {total_volume_frac / total_parents:.1f} %')
    if save_file:
        array_to_csv(path=parent_name, arr=parents, dtype=int, mode='w', save_as_int=True)
    return parents


if __name__ == '__main__':
    number_of_voxels_x = 5
    number_of_voxels_y = 5
    number_of_voxels_z = 5
    number_of_parents_to_generate = 5
    parent_cell_density = 0.2
    probability_of_mutation_in_structure = 0.05
    addition_of_voxels_in_validation_process_probability = 0.01
    timeout_of_validation_process = 0.5

    topology_2d = random_parents_generation(lx=number_of_voxels_x, ly=number_of_voxels_y, lz=number_of_voxels_z,
                                            total_parents=number_of_parents_to_generate, density=parent_cell_density,
                                            mutation_probability=probability_of_mutation_in_structure,
                                            add_probability=addition_of_voxels_in_validation_process_probability,
                                            timeout=timeout_of_validation_process,
                                            save_file=True)
    for i in range(number_of_parents_to_generate):
        visualize_one_cube(topology_2d[i].reshape((number_of_voxels_x, number_of_voxels_y, number_of_voxels_z)))
    offsprings = generate_offspring(topologies=topology_2d, w=1, end_pop=number_of_parents_to_generate,
                                    mutation_rate=probability_of_mutation_in_structure,
                                    add_probability=addition_of_voxels_in_validation_process_probability,
                                    timeout=timeout_of_validation_process,
                                    lx=number_of_voxels_x, ly=number_of_voxels_y, lz=number_of_voxels_z)
    for i in range(number_of_parents_to_generate):
        visualize_one_cube(offsprings[i])
