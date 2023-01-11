import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def get_datum_hv(pareto_1_sorted, pareto_2_sorted):
    datum_point_x, datum_point_y = pareto_1_sorted[-1], pareto_2_sorted[0]
    xrs = pareto_1_sorted - datum_point_x
    yrs = pareto_2_sorted - datum_point_y
    datum_hv = 0
    for point_idx, (xr, yr) in enumerate(zip(xrs, yrs)):
        if point_idx == len(xrs) - 1:
            break
        datum_hv += abs((xrs[point_idx + 1] - xr) * (yrs[point_idx + 1] + yr) * 0.5)
    datum_hv -= abs((pareto_1_sorted[-1]-pareto_1_sorted[0])*(pareto_2_sorted[0]-pareto_2_sorted[-1]))
    return datum_hv


def get_hv_from_datum_hv(datum_hv, lower_bounds, ref_x, ref_y):
    return datum_hv + (ref_x - lower_bounds[0]) * (ref_y - lower_bounds[1])


def evaluate_fitness_values(topo, result, params):
    fitness_values = np.empty_like(result)
    max_rf22 = params.MaxRF22
    lx, ly, lz = params.lx, params.ly, params.lz
    k = params.penalty_coefficient
    if params.evaluation_version == 'ver1':
        for i in range(topo.shape[0]):
            dis11 = result[i, 0]
            dis22 = result[i, 1]
            rf22 = result[i, 4]
            fit_val1 = (rf22 / max_rf22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i, 0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i, 1] = fit_val2
    elif params.evaluation_version == 'ver2':
        for i in range(topo.shape[0]):
            rf22 = result[i, 4]
            fit_val1 = np.sum(topo[i]) / (lx * ly * lz)
            fitness_values[i, 0] = fit_val1
            fit_val2 = rf22 / max_rf22
            fitness_values[i, 1] = fit_val2
    elif params.evaluation_version == 'ver3':
        for i in range(topo.shape[0]):
            dis11 = result[i, 0]
            dis22 = result[i, 1]
            dis33 = result[i, 2]
            fit_val1 = - (dis11 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i, 0] = fit_val1
            fit_val2 = - (dis33 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i, 1] = fit_val2
    elif params.evaluation_version == 'ver4':
        for i in range(topo.shape[0]):
            fit_val1 = result[i, 9]
            fitness_values[i, 0] = fit_val1
            fit_val2 = np.sum(topo[i]) / (lx * ly * lz)
            fitness_values[i, 1] = fit_val2
    return fitness_values


# def get_fitness_value_limits(w, lx, ly, lz, evaluation_version, penalty_coefficient, max_rf22):
#     fit_min, fit_max = 0, 0
#     for i in range(1, w + 1):
#         topo, result = parent_import(w=i)
#         fitness_values = evaluate_one_topology(topo=topo, result=result, lx=lx, ly=ly, lz=lz,
#                                                evaluation_version=evaluation_version,
#                                                max_rf22=max_rf22, penalty_coefficient=penalty_coefficient)
#         if i == 1:
#             fit_max = np.max(fitness_values, axis=0)
#             fit_min = np.min(fitness_values, axis=0)
#         else:
#             fit_max1 = np.max(fitness_values, axis=0)
#             fit_min1 = np.min(fitness_values, axis=0)
#             for j in range(2):
#                 if fit_max1[j] > fit_max[j]:
#                     fit_max[j] = fit_max1[j]
#                 if fit_min1[j] < fit_min[j]:
#                     fit_min[j] = fit_min1[j]
#     return fit_max, fit_min


# def fitval_sort(fv):
#     fv_sp = np.split(fv, 2, axis=1)
#     fv1 = fv_sp[0]
#     fv2 = fv_sp[1]
#     fv1 = np.squeeze(fv1)
#     fv2 = np.squeeze(fv2)
#     sort = fv1.argsort()
#     fv1_sort = fv1[sort]
#     fv2_sort = fv2[sort]
#     return fv1_sort, fv2_sort, sort


def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)  # initially assume all solutions are in pareto front by using "1"

    for i in range(pop_size):
        for j in range(pop_size):
            if np.less_equal(fitness_values[j], fitness_values[i]).all() and np.less(fitness_values[j],
                                                                                     fitness_values[i]).any():
                pareto_front[i] = 0  # i is not in pareto front because j dominates i
                break  # no more comparision is needed to find out which one is dominated

    return pop_index[pareto_front]


# def visualize_old(w, lx, ly, lz, penalty_coefficient, evaluation_version, max_rf22, parent_conn,
#               file_io=True):
# print('\n')
# print("iteration: %d" % w)
# print("_________________")
# print("Optimal solutions:")
# print("       x1               x2                 x3")
# print(index)  # show optimal solutions
# print("______________")
# print("Fitness values:")
# # print("  objective 1    objective 2")
# print("          Model              objective 1      objective 2      Job")
# print("            |                     |                |            |")
# for q in range(len(index)):
#     gen_num, pop_num = find_model_output(w=w+1, q=sort[q], directory=directory, end_pop=end_pop)
#     print("%dth generation %dth pop   [%E   %E]   Job-%d-%d" % (
#         w + 1, q + 1, fitness_values[q, 0], fitness_values[q, 1], gen_num, pop_num))
# print(fitness_values)

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


def crowding_calculation(fitness_values):
    pop_size = len(fitness_values[:, 0])
    fitness_value_number = len(fitness_values[0, :])
    matrix_for_crowding = np.zeros((pop_size, fitness_value_number))
    normalize_fitness_values = (fitness_values - fitness_values.min(0)) / fitness_values.ptp(0)

    # normalize the fitness values
    for i in range(fitness_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1  # extreme point has the max crowding distance
        crowding_results[pop_size - 1] = 1  # extreme point has the max crowding distance
        sorting_normalize_fitness_values = np.sort(normalize_fitness_values[:, i])
        sorting_normalized_values_index = np.argsort(normalize_fitness_values[:, i])

        # crowding distance calculation
        crowding_results[1:pop_size - 1] = (
                sorting_normalize_fitness_values[2:pop_size] - sorting_normalize_fitness_values[0:pop_size - 2])
        re_sorting = np.argsort(sorting_normalized_values_index)  # re_sorting to the original order
        matrix_for_crowding[:, i] = crowding_results[re_sorting]

    crowding_distance = np.sum(matrix_for_crowding, axis=1)  # crowding distance of each solution
    return crowding_distance


def remove_using_crowding(fitness_values, number_solutions_needed):
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


def selection(all_topologies, all_fitness_values, population_size):  # More efficient algorithm
    total_pop_idx = set(np.arange(len(all_topologies)))
    remaining_population_idx = np.arange(len(all_topologies))
    total_pareto_front_idx = set()
    while len(total_pareto_front_idx) < population_size:
        new_pareto_front_idx = find_pareto_front_points(costs=all_fitness_values[remaining_population_idx],
                                                        return_index=True)
        total_pareto_size = len(total_pareto_front_idx) + len(new_pareto_front_idx)

        # check the size of pareto front, if larger than self.population_size,
        # remove some solutions using crowding criterion
        if total_pareto_size > population_size:
            number_solutions_needed = population_size - len(total_pareto_front_idx)
            selected_solutions = remove_using_crowding(all_fitness_values[new_pareto_front_idx],
                                                       number_solutions_needed)
            new_pareto_front_idx = new_pareto_front_idx[selected_solutions]

        total_pareto_front_idx = total_pareto_front_idx.union(remaining_population_idx[new_pareto_front_idx])
        remaining_population_idx = np.array(list(total_pop_idx - total_pareto_front_idx))
    selected_populations = all_topologies[list(total_pareto_front_idx)]
    return selected_populations, list(total_pareto_front_idx)


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


if __name__ == '__main__':
    pass
