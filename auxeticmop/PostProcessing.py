import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from .ClassDefinitions import Parameters


def get_datum_hv(pareto_1_sorted: np.ndarray, pareto_2_sorted: np.ndarray) -> float:
    datum_point_x, datum_point_y = pareto_1_sorted[-1], pareto_2_sorted[0]
    xrs = pareto_1_sorted - datum_point_x
    yrs = pareto_2_sorted - datum_point_y
    datum_hv = 0
    for point_idx, (xr, yr) in enumerate(zip(xrs, yrs)):
        if point_idx == len(xrs) - 1:
            break
        datum_hv += abs((xrs[point_idx + 1] - xr) * (yrs[point_idx + 1] + yr) * 0.5)
    datum_hv -= abs((pareto_1_sorted[-1] - pareto_1_sorted[0]) * (pareto_2_sorted[0] - pareto_2_sorted[-1]))
    return datum_hv


def get_hv_from_datum_hv(datum_hv: float, lower_bounds: list, ref_x: float, ref_y: float):
    return datum_hv + (ref_x - lower_bounds[0]) * (ref_y - lower_bounds[1])


def evaluate_fitness_values(topo: np.ndarray[int], result: dict, params: Parameters) -> np.ndarray[float]:
    fitness_values = np.empty((len(topo), 2), dtype=float)
    max_rf22 = params.MaxRF22
    lx, ly, lz = params.lx, params.ly, params.lz
    k = params.penalty_coefficient
    if params.evaluation_version == 'ver1':
        for offspring_idx in range(topo.shape[0]):
            dis11 = result[offspring_idx, 0]
            dis22 = result[offspring_idx, 1]
            rf22 = result[offspring_idx, 4]
            fit_val1 = (rf22 / max_rf22) + k * (np.sum(topo[offspring_idx]) / (lx * ly * lz))
            fitness_values[offspring_idx, 0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo[offspring_idx]) / (lx * ly * lz))
            fitness_values[offspring_idx, 1] = fit_val2
    elif params.evaluation_version == 'ver2':
        for offspring_idx in range(topo.shape[0]):
            rf22 = result[offspring_idx, 4]
            fit_val1 = np.sum(topo[offspring_idx]) / (lx * ly * lz)
            fitness_values[offspring_idx, 0] = fit_val1
            fit_val2 = rf22 / max_rf22
            fitness_values[offspring_idx, 1] = fit_val2
    elif params.evaluation_version == 'ver3':
        for offspring_idx in range(len(result)):
            dis11 = result[offspring_idx+1]['displacement']['xMax'][0]
            dis22 = result[offspring_idx+1]['displacement']['yMax'][1]
            dis33 = result[offspring_idx+1]['displacement']['zMax'][2]
            fit_val1 = - (dis11 / dis22) + k * (np.sum(topo[offspring_idx]) / (lx * ly * lz))
            fitness_values[offspring_idx, 0] = fit_val1
            fit_val2 = - (dis33 / dis22) + k * (np.sum(topo[offspring_idx]) / (lx * ly * lz))
            fitness_values[offspring_idx, 1] = fit_val2
    elif params.evaluation_version == 'ver4':
        for offspring_idx in range(topo.shape[0]):
            fit_val1 = result[offspring_idx, 9]
            fitness_values[offspring_idx, 0] = fit_val1
            fit_val2 = np.sum(topo[offspring_idx]) / (lx * ly * lz)
            fitness_values[offspring_idx, 1] = fit_val2
    return fitness_values


def find_pareto_front_points(costs: np.ndarray, return_index: bool = False) -> np.ndarray:
    """
    A function calculating indices of pareto fronts or values of pareto points. The returned value is sorted by the
    first cost value.
    :param costs: The array containing fitness values, shape: (no_of_points x no_of_costs)
    :param return_index: If True, returns indices of pareto fronts. If False, returns fitness values of pareto fronts.
    :return: if return_index == True, The list of indices of pareto front points from costs, len: no_of_pareto_points
    if return_index == False, The array containing fitness values of pareto points, shape: (no_of_pareto_front_points
    x no_of_costs)
    """
    costs_copy = costs.copy()
    unsorted_indices = np.arange(len(costs_copy))
    next_point_idx = 0
    while next_point_idx < len(costs_copy):
        not_dominated_point_mask = np.any(costs_copy < costs_copy[next_point_idx], axis=1)
        not_dominated_point_mask[next_point_idx] = True
        unsorted_indices = unsorted_indices[not_dominated_point_mask]
        costs_copy = costs_copy[not_dominated_point_mask]
        next_point_idx = np.sum(not_dominated_point_mask[:next_point_idx]) + 1
    sorted_indices = unsorted_indices[np.argsort(costs[unsorted_indices][:, 0])]
    sorted_pareto_points = costs[sorted_indices]
    if return_index:
        return sorted_indices
    else:
        return sorted_pareto_points


def crowding_calculation(fitness_values: np.ndarray):
    print('fitval: ', fitness_values)
    population_size, number_of_fitness_values = fitness_values.shape[0], fitness_values.shape[1]
    matrix_for_crowding = np.zeros((population_size, number_of_fitness_values))
    normalize_fitness_values = (fitness_values - np.min(fitness_values, axis=0)) / np.ptp(fitness_values, axis=0)

    # normalize the fitness values
    for i in range(number_of_fitness_values):
        crowding_results = np.zeros(population_size)
        crowding_results[0] = 1  # extreme point has the max crowding distance
        crowding_results[population_size - 1] = 1  # extreme point has the max crowding distance
        sorting_normalize_fitness_values = np.sort(normalize_fitness_values[:, i])
        sorting_normalized_values_index = np.argsort(normalize_fitness_values[:, i])

        # crowding distance calculation
        crowding_results[1:population_size - 1] = (sorting_normalize_fitness_values[2:population_size]
                                                   - sorting_normalize_fitness_values[0:population_size - 2])
        re_sorting = np.argsort(sorting_normalized_values_index)  # re_sorting to the original order
        matrix_for_crowding[:, i] = crowding_results[re_sorting]

    crowding_distance = np.sum(matrix_for_crowding, axis=1)  # crowding distance of each solution
    print('crowding distance: ', crowding_distance)
    return crowding_distance


def remove_using_crowding(fitness_values: np.ndarray, number_solutions_needed: int) -> np.ndarray:
    rn = np.random  # addition
    pop_index = np.arange(fitness_values.shape[0])
    crowding_distance = crowding_calculation(fitness_values)
    selected_pop_index = np.zeros(number_solutions_needed)
    selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))

    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            # solution 1 is better than solution 2
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, solution_1, axis=0)  # remove the selected solution
            fitness_values = np.delete(fitness_values, solution_1,
                                       axis=0)  # remove the fitness of the selected solution
            crowding_distance = np.delete(crowding_distance, solution_1,
                                          axis=0)  # remove the related crowding distance

        else:
            # solution 2 is better than solution 1
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, solution_2, axis=0)
            fitness_values = np.delete(fitness_values, solution_2, axis=0)
            crowding_distance = np.delete(crowding_distance, solution_2, axis=0)

    selected_pop_index = np.asarray(selected_pop_index, dtype=int)  # Convert the data to integer
    return selected_pop_index


def selection(all_fitness_values: np.ndarray, selected_size: int) -> np.ndarray:
    remaining_population_idx = np.arange(len(all_fitness_values))
    pareto_indices = np.empty((0,), dtype=int)
    while len(pareto_indices) < selected_size:
        new_pareto_front_idx = find_pareto_front_points(costs=all_fitness_values[remaining_population_idx],
                                                        return_index=True)
        total_pareto_size = len(pareto_indices) + len(new_pareto_front_idx)

        # check the size of pareto front, if larger than self.population_size,
        # remove some solutions using crowding criterion
        if total_pareto_size > selected_size:
            number_solutions_needed = selected_size - len(pareto_indices)
            selected_solutions = remove_using_crowding(all_fitness_values[new_pareto_front_idx],
                                                       number_solutions_needed)
            new_pareto_front_idx = new_pareto_front_idx[selected_solutions]
        pareto_indices = np.hstack((pareto_indices, remaining_population_idx[new_pareto_front_idx]))
        remaining_population_idx = np.setdiff1d(remaining_population_idx, pareto_indices)
    return pareto_indices


def array_divide(topo, lx, ly, lz, divide_number, ini_pop, end_pop):
    lxx = lx / divide_number
    lyy = ly / divide_number
    lzz = lz / divide_number
    topo = topo.reshape((end_pop, lzz, lyy, lxx))
    topo_divided = np.zeros((end_pop, lz, ly, lx))
    for q in range(ini_pop, end_pop + 1):
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    topo_divided[q - 1][k][j][i] = topo[q - 1][k // divide_number][j // divide_number][
                        i // divide_number]
    return topo_divided


def filter_process(topo_divided, sigma, threshold, lx, ly, lz, ini_pop, end_pop):
    topo_filtered = np.zeros((end_pop, lz, ly, lx))
    for q in range(ini_pop, end_pop):
        topo_divided2 = gaussian_filter(topo_divided[q - 1], sigma=sigma)
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    if topo_divided2[k][j][i] >= threshold:
                        topo_divided2[k][j][i] = 1
                    else:
                        topo_divided2[k][j][i] = 0
        topo_filtered[q - 1] = topo_divided2
    return topo_filtered


def quaver_to_full(quaver):
    quarter = np.concatenate((np.flip(quaver, axis=0), quaver), axis=0)
    half = np.concatenate((np.flip(quarter, axis=1), quarter), axis=1)
    full = np.concatenate((np.flip(half, axis=2), half), axis=2)
    return np.swapaxes(full, axis1=0, axis2=2)


def visualize_one_cube(cube_3d_array, full=False):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={
        'projection': '3d'
    })
    ax.set(ylabel='y', xticks=[], zticks=[])
    ax.voxels(quaver_to_full(cube_3d_array) if full else cube_3d_array, facecolor='#e02050', edgecolors='k')
    ax.grid(True)
    plt.show()


def visualize_n_cubes(arr_4d, full=False):
    n_row = 1
    n_col = arr_4d.shape[0]
    fig, ax = plt.subplots(n_row, n_col, figsize=(20, 5), subplot_kw={
        'projection': '3d'
    })
    for idx in range(n_col):
        ax[idx].set(xlabel='x', ylabel='y', zlabel='z')
        ax[idx].voxels(quaver_to_full(arr_4d[idx]) if full else arr_4d[idx],
                       facecolor='r', edgecolors='k')
        ax[idx].grid(True)
    plt.show()

#
# def show_pareto_fronts(gen: int, params, directory: str = None, show: bool = False) -> np.ndarray:
#     def find_job_location_from_offspring(g, tp):
#         for current_gen in range(g, 0, -1):
#             topos_offspring = np.genfromtxt(f'topo_offspring_{current_gen}.csv', dtype=int, delimiter=',')
#             for offspring_idx, topo_offspring in enumerate(topos_offspring):
#                 if np.array_equal(tp, topo_offspring):
#                     return current_gen, offspring_idx + 1
#         topos_parent_1 = np.genfromtxt(f'topo_parent_1.csv', dtype=int, delimiter=',')
#         for offspring_idx, topo_parent_1 in enumerate(topos_parent_1):
#             if np.array_equal(tp, topo_parent_1):
#                 return 0, offspring_idx + 1
#
#     if directory is not None:
#         from os import chdir
#         chdir(directory)
#     topos_next_parent = np.genfromtxt(f'topo_parent_{gen + 1}.csv', dtype=int, delimiter=',')
#     results_next_parent = np.genfromtxt(f'Output_parent_{gen + 1}.csv', dtype=float, delimiter=',')
#
#     fitness_values = evaluate_fitness_values(topo=topos_next_parent, result=results_next_parent, params=params)
#     pareto_indices = find_pareto_front_points(costs=fitness_values, return_index=True)
#     print('=' * 66)
#     for pareto_idx in pareto_indices:
#         topo_pareto = topos_next_parent[pareto_idx]
#         job_first, job_second = find_job_location_from_offspring(g=gen, tp=topo_pareto)
#         print(f'[Gen {gen}]', f'Parent {gen + 1}-{pareto_idx + 1}:',
#               f'{fitness_values[pareto_idx, 0]:.10f} |',
#               f'{fitness_values[pareto_idx, 1]:.10f} |',
#               f'Job{job_first}-{job_second}.odb')
#     if show:
#         pareto_topologies = topos_next_parent[pareto_indices].reshape(len(pareto_indices),
#                                                                       params.lx, params.ly, params.lz)
#         for pareto_topology in pareto_topologies:
#             visualize_one_cube(cube_3d_array=pareto_topology, full=True)
#     return pareto_indices


def print_origin_of_paretos(params) -> None:
    """
    Print original location of pareto topologies from whole csv files.
    :param params: Parameter dataclass
    :return: None
    """
    from os import listdir
    from re import compile
    from threading import Thread

    def _get_numbers(p):
        return sorted([int(compile(r'\d+').search(s).group()) for s in [f for f in listdir() if compile(p).match(f)]])

    def _load_whole_files(header, nd, ad):
        if 'topo' in header:
            dtype = int
        else:
            dtype = float
        nd.update({header: _get_numbers(rf'{header}_\d+\.csv')})
        ad.update({header: np.expand_dims(np.genfromtxt(f'{header}_{nd[header][0]}.csv',
                                                        dtype=dtype, delimiter=','), axis=0)})
        for file_idx, num in enumerate(nd[header]):
            if file_idx == 0:
                continue
            ad[header] = np.vstack((ad[header],
                                    np.expand_dims(np.genfromtxt(f'{header}_{num}.csv',
                                                                 dtype=dtype, delimiter=','), axis=0)))
        return ad

    def _find_job_location_from_offspring(gen, pareto_idx, tpp, tpo, tpfp, nd):
        # tpp: topo_pareto, tpo: topo_offspring, tpfp: topo_first_parent, nd: num_dict
        arg = np.argwhere(np.all(tpo == tpp, axis=(2,)))

        if len(arg) == 0:
            arg = np.argwhere(np.all(tpfp == tpp, axis=(1,)))
            print(f'Pareto topo in parent {gen} - {pareto_idx + 1} is in parent ', 1, '-', arg[0, 0] + 1)
        else:
            print(f'Pareto topo in parent {gen} - {pareto_idx + 1} is in offspring ', nd['topo_offspring'][arg[0, 0]],
                  '-', arg[0, 1] + 1)

    filename_headers = ('topo_parent', 'topo_offspring', 'Output_parent', 'Output_offspring')
    num_dict, arr_dict = dict(), dict()
    threads = [Thread(target=_load_whole_files, args=(header, num_dict, arr_dict), daemon=True)
               for header in filename_headers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    del threads
    for arr_idx, gen in enumerate(num_dict['Output_parent']):
        fitness_values = evaluate_fitness_values(topo=arr_dict['topo_parent'][arr_idx],
                                                 result=arr_dict['Output_parent'][arr_idx], params=params)
        pareto_indices = find_pareto_front_points(costs=fitness_values, return_index=True)
        print('=' * 30, f'Parent {gen}', '=' * 30)
        for pareto_idx in pareto_indices:
            topo_pareto = arr_dict['topo_parent'][arr_idx, pareto_idx]
            _find_job_location_from_offspring(gen=gen, pareto_idx=pareto_idx, tpp=topo_pareto, nd=num_dict,
                                              tpo=arr_dict['topo_offspring'], tpfp=arr_dict['topo_parent'][0])


def open_history_output(gen, path=None):
    if path is not None:
        from os import chdir
        chdir(path)
    from pickle import load
    file_names = ('U1_HistoryOutput', 'U2_HistoryOutput', 'U3_HistoryOutput',
                  'RF1_HistoryOutput', 'RF2_HistoryOutput', 'RF3_HistoryOutput')
    for file_name in file_names:
        pickle_file_name = file_name + f'_{gen}'
        with open(pickle_file_name, 'rb') as f:
            loaded_file = load(f, encoding='bytes')
            print(f'Gen{gen}-{pickle_file_name}: {loaded_file}')


if __name__ == '__main__':
    from GraphicUserInterface import Parameters
    from os import chdir

    set_path = r'f:\shshsh\data-23-1-4'
    chdir(set_path)
    parameters = Parameters()
    parameters.post_initialize()
    print_origin_of_paretos(params=parameters)
