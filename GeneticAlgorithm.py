import random
import numpy as np
import itertools
import os
from functools import reduce
from MutateAndValidate import mutate_and_validate_topology, visualize_one_cube
from FileIO import array_to_csv


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
    cuttings = np.arange(0, topologies.shape[1], step=1, dtype=int)
    np.random.shuffle(cuttings)
    cutting, candidate = None, None
    candidate_found_flag = False
    while not candidate_found_flag:
        candidate = list()
        cutting = cuttings[-1]
        cuttings = np.delete(cuttings, obj=-1, axis=0)
        for parent_idx in range(len(topologies)):
            if topologies[parent_idx, cutting] == 1:
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
    offspring1 = np.zeros_like(chromosome_1)
    offspring2 = np.zeros_like(chromosome_2)
    offspring1[0:cutting_section] = chromosome_1[0:cutting_section]
    offspring1[cutting_section:] = chromosome_2[cutting_section:]
    offspring2[0:cutting_section] = chromosome_2[0:cutting_section]
    offspring2[cutting_section:] = chromosome_1[cutting_section:]
    offspring = np.vstack([offspring1, offspring2])
    return offspring


def generate_offspring(topologies, w, lx, ly, lz, end_pop, mutation_rate, timeout):
    offspring = np.empty((0, lx * ly * lz), int)
    trial = 1
    validation_count = 0
    all_parents_topologies = np.empty((0, end_pop, lx * ly * lz), int)
    for generation_idx in range(w):
        parent_topologies = np.genfromtxt('topo_parent_' + str(generation_idx + 1) + '.csv', delimiter=',',
                                          dtype=int).reshape((1, end_pop, lx * ly * lz))
        all_parents_topologies = np.vstack((all_parents_topologies, parent_topologies))
    while True:
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
                                                                  mutation_probability=mutation_rate, timeout=timeout)
            validated_chromosome_2 = mutate_and_validate_topology(cross_overed_chromosome_2,
                                                                  mutation_probability=mutation_rate, timeout=timeout)
            for validated_chromosome in (validated_chromosome_1, validated_chromosome_2):
                if validated_chromosome is None:
                    print('[Generate offspring] <!> Non-connected tree detected')
                    continue
                else:
                    is_any_clone_in_all_parents = inspect_clone_in_all_parents(
                        w=w, topology_flattened=validated_chromosome, all_parent_topologies=all_parents_topologies)
                    is_any_clone_in_current_offsprings = inspect_clone_in_current_offsprings(
                        topology=validated_chromosome, offspring=offspring)
                    if is_any_clone_in_all_parents:
                        print('[Generate offspring] Clone structure found in parents!')
                        continue
                    elif is_any_clone_in_current_offsprings:
                        print('[Generate offspring] Clone structure found in current offsprings!')
                        continue
                    else:
                        offspring = np.vstack((offspring, validated_chromosome.flatten()))
                        validation_count += 1
                        print(f'[Generate offspring] Validation of chromosome {validation_count} complete!')
                        if len(offspring) == end_pop:
                            print('[Generate offspring] Generating offspring complete')
                            array_to_csv(f'topo_offspring_{w}.csv', offspring, dtype=int, mode='w', save_as_int=True)
                            return offspring.reshape((end_pop, lx, ly, lz))


def random_array(shape, probability):
    return np.random.choice([1, 0], size=reduce(lambda x, y: x * y, shape), p=[probability, 1 - probability]).reshape(
        shape)


def random_parent_generation(lx, ly, lz, total_offsprings, density, mutation_probability, timeout,
                             save_file=True):
    parent_name = 'topo_parent_1.csv'
    parents = np.empty((total_offsprings, lx * ly * lz))
    total_parent_generation_count = 0
    total_volume_frac = 0
    while total_parent_generation_count < total_offsprings:
        rand_arr = random_array(shape=(lx, ly, lz), probability=density)
        parent = mutate_and_validate_topology(rand_arr, mutation_probability=mutation_probability,
                                              timeout=timeout)
        if parent is None:
            continue
        total_parent_generation_count += 1
        print(f'<<<<< Parent {total_parent_generation_count + 1} >>>>>')

        volume_frac = np.count_nonzero(parent) / (lx * ly * lz / 100)
        total_volume_frac += volume_frac
        print(f'Volume fraction: {volume_frac:.1f} %\n')
        parents[total_parent_generation_count] = parent.flatten()
    print(f'Average volume fraction: {total_volume_frac / total_offsprings:.1f} %')
    if save_file:
        array_to_csv(path=parent_name, arr=parents, dtype=int, mode='w', save_as_int=True)
    else:
        for parent in parents:
            visualize_one_cube(parent.reshape((lx, ly, lz)), full=False)


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
    topos, _ = parent_import(w=18, restart_pop=0)  # topo: (100, 1000), result: (100, 12)
    number_of_voxels_x = 10
    number_of_voxels_y = 10
    number_of_voxels_z = 10
    end_population = 100
    offsprings = generate_offspring(topologies=topos, w=18, end_pop=end_population,
                                    mutation_rate=0.05, timeout=0.5,
                                    lx=number_of_voxels_x, ly=number_of_voxels_y, lz=number_of_voxels_z)
    inspect_topologies(generation=19)
    for off_idx in range(100):
        visualize_one_cube(offsprings[off_idx])
