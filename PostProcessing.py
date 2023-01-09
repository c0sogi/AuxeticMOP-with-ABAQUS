import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
from FileIO import parent_import

np.set_printoptions(linewidth=np.inf)


def evaluation(topo, topo_1, reslt, reslt_1, q, lx, ly, lz, evaluation_version, penalty_coefficient, max_rf22):
    fitness_values = np.zeros((2 * q, 2))
    k = penalty_coefficient
    if evaluation_version == 'ver1':
        for i in range(q):
            dis11 = reslt[i][0]
            dis22 = reslt[i][1]
            rf22 = reslt[i][4]
            fit_val1 = (rf22 / max_rf22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

        for i in range(q, 2 * q):
            dis11 = reslt_1[i - q][0]
            dis22 = reslt_1[i - q][1]
            rf22 = reslt_1[i - q][4]
            fit_val1 = (rf22 / max_rf22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver2':
        for i in range(q):
            rf22 = reslt[i][4]
            fit_val1 = np.sum(topo[i]) / (lx * ly * lz)
            fitness_values[i][0] = fit_val1
            fit_val2 = rf22 / max_rf22
            fitness_values[i][1] = fit_val2

        for i in range(q, 2 * q):
            rf22 = reslt_1[i - q][4]
            fit_val1 = np.sum(topo_1[i - q]) / (lx * ly * lz)
            fitness_values[i][0] = fit_val1
            fit_val2 = rf22 / max_rf22
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver3':
        for i in range(q):
            dis11 = reslt[i][0]
            dis22 = reslt[i][1]
            dis33 = reslt[i][2]
            fit_val1 = - (dis11 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis33 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

        for i in range(q, 2 * q):
            print('reslt1', reslt_1, reslt_1.shape)
            dis11 = reslt_1[i - q][0]
            dis22 = reslt_1[i - q][1]
            dis33 = reslt_1[i - q][2]
            fit_val1 = - (dis11 / dis22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis33 / dis22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver4':
        for i in range(q):
            fit_val1 = reslt[i][9]
            fitness_values[i][0] = fit_val1
            fit_val2 = np.sum(topo[i]) / (lx * ly * lz)
            fitness_values[i][1] = fit_val2
        for i in range(q, 2 * q):
            fit_val1 = reslt_1[i - q][9]
            fitness_values[i][0] = fit_val1
            fit_val2 = np.sum(topo_1[i - q]) / (lx * ly * lz)
            fitness_values[i][1] = fit_val2
    return fitness_values


def evaluation2(topo2, reslt2, lx, ly, lz, penalty_coefficient, evaluation_version, max_rf22):
    fitness_values = np.zeros((topo2.shape[0], 2))
    k = penalty_coefficient
    if evaluation_version == 'ver1':
        for i in range(topo2.shape[0]):
            dis11 = reslt2[i][0]
            dis22 = reslt2[i][1]
            rf22 = reslt2[i][4]
            fit_val1 = (rf22 / max_rf22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver2':
        for i in range(topo2.shape[0]):
            rf22 = reslt2[i][4]
            fit_val1 = np.sum(topo2[i]) / (lx * ly * lz)
            fitness_values[i][0] = fit_val1
            fit_val2 = rf22 / max_rf22
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver3':
        for i in range(topo2.shape[0]):
            dis11 = reslt2[i][0]
            dis22 = reslt2[i][1]
            dis33 = reslt2[i][2]
            fit_val1 = - (dis11 / dis22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis33 / dis22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver4':
        for i in range(topo2.shape[0]):
            fit_val1 = reslt2[i][9]
            fitness_values[i][0] = fit_val1
            fit_val2 = np.sum(topo2[i]) / (lx * ly * lz)
            fitness_values[i][1] = fit_val2

    return fitness_values


def max_fitval(w, lx, ly, lz, evaluation_version, penalty_coefficient, max_rf22):
    fit_min, fit_max = 0, 0
    for i in range(1, w + 1):
        topo, result = parent_import(w=i, restart_pop=0)
        fitness_values = evaluation2(topo2=topo, reslt2=result, lx=lx, ly=ly, lz=lz,
                                     evaluation_version=evaluation_version,
                                     max_rf22=max_rf22, penalty_coefficient=penalty_coefficient)
        if i == 1:
            fit_max = np.max(fitness_values, axis=0)
            fit_min = np.min(fitness_values, axis=0)
        else:
            fit_max1 = np.max(fitness_values, axis=0)
            fit_min1 = np.min(fitness_values, axis=0)
            for j in range(2):
                if fit_max1[j] > fit_max[j]:
                    fit_max[j] = fit_max1[j]
                if fit_min1[j] < fit_min[j]:
                    fit_min[j] = fit_min1[j]
    return fit_max, fit_min


def fitval_sort(fv):
    fv_sp = np.split(fv, 2, axis=1)
    fv1 = fv_sp[0]
    fv2 = fv_sp[1]
    fv1 = np.squeeze(fv1)
    fv2 = np.squeeze(fv2)
    sort = fv1.argsort()
    fv1_sort = fv1[sort]
    fv2_sort = fv2[sort]
    return fv1_sort, fv2_sort, sort


def hypervolume_calculation(fv1_sort, fv2_sort, w, lx, ly, lz, evaluation_version,
                            penalty_coefficient, max_rf22):
    fit_max, fit_min = max_fitval(w=w, lx=lx, ly=ly, lz=lz,
                                  evaluation_version=evaluation_version, penalty_coefficient=penalty_coefficient,
                                  max_rf22=max_rf22)
    rp_max1 = fit_max[0] + 0.05 * (fit_max[0] - fit_min[0])
    rp_min1 = fit_min[0] - 0.05 * (fit_max[0] - fit_min[0])
    rp_max2 = fit_max[1] + 0.05 * (fit_max[1] - fit_min[1])
    rp_min2 = fit_min[1] - 0.05 * (fit_max[1] - fit_min[1])
    hv = 0
    area = (rp_max1 - rp_min1) * (rp_max2 - rp_min2)
    pareto_no = len(fv1_sort)
    for i in range(pareto_no):
        if i == 0:
            hv += (fv1_sort[i] - rp_min1) * (rp_max2 - fv2_sort[i]) + (rp_max1 - fv1_sort[pareto_no - 1 - i]) * (
                    rp_max2 - fv2_sort[pareto_no - 1 - i])
        else:
            hv += ((fv1_sort[i] - fv1_sort[i - 1]) * (rp_max2 - fv2_sort[i]) + (
                    fv1_sort[pareto_no - i] - fv1_sort[pareto_no - 1 - i]) * (
                           rp_max2 - fv2_sort[pareto_no - 1 - i])) * 0.5
    normalized_hv = hv / area
    return normalized_hv, hv


def hypervolume_calculation2(fv1_sort, fv2_sort):
    pareto_no = len(fv1_sort)
    hv = 0
    for idx in range(pareto_no - 1):
        hv += (fv1_sort[idx + 1] - fv1_sort[idx]) * (-fv2_sort[idx])
    hv += fv1_sort[pareto_no - 1] * fv2_sort[pareto_no - 1]
    return hv


def hypervolume_calculation3(fv1_sort, fv2_sort, ref_x, ref_y):
    x_relative = fv1_sort - ref_x
    y_relative = fv2_sort - ref_y
    x_integral_lower_limit = fv1_sort[0] - ref_x
    x_integral_upper_limit = fv1_sort[-1] - ref_x
    right_gap = ref_x - fv1_sort[-1]
    right_height = -fv2_sort[-1]
    interpolation_coefficient = 1  # 1: Linear interpolation

    f = InterpolatedUnivariateSpline(x_relative, y_relative, k=interpolation_coefficient)
    hv = quad(lambda x: abs(f(x)), x_integral_lower_limit, x_integral_upper_limit)[0] + abs(right_gap * right_height)
    return hv


# def find_model_output(w, q, directory, end_pop):
#     no_of_datafile = w
#     no_of_lines = end_pop
#     if directory[-1] != '/' or directory[-1] != '\\':
#         directory += '/'
#
#     f = open(directory + 'topo_parent_%d.csv' % w, 'r')
#     rd = csv.reader(f)
#     topo = [i for i in rd]
#     topo = np.array(topo)
#     topo = topo.astype(np.float32)
#     gen_num_eng = 0
#     pop_num_eng = 0
#     f = open(directory + 'Output_parent_%d.csv' % w, 'r')
#     rd = csv.reader(f)
#     eng = [i for i in rd]
#     eng = np.array(eng)
#     eng = eng.astype(np.float32)
#
#     for j in range(no_of_datafile):
#         break1 = 0
#         if j == 0:
#             f = open(directory + 'topo_parent_%d.csv' % (j + 1), 'r')
#             rd = csv.reader(f)
#             topo1 = [i for i in rd]
#             topo1 = np.array(topo1)
#             topo1 = topo1.astype(np.float32)
#             f = open(directory + 'Output_parent_%d.csv' % (j + 1), 'r')
#             rd = csv.reader(f)
#             eng1 = [i for i in rd]
#             eng1 = np.array(eng1)
#             eng1 = eng1.astype(np.float32)
#         else:
#             f = open(directory + 'topo_offspring_%d.csv' % j, 'r')
#             rd = csv.reader(f)
#             topo1 = [i for i in rd]
#             topo1 = np.array(topo1)
#             topo1 = topo1.astype(np.float32)
#             f = open(directory + 'Output_offspring_%d.csv' % j, 'r')
#             rd = csv.reader(f)
#             eng1 = [i for i in rd]
#             eng1 = np.array(eng1)
#             eng1 = eng1.astype(np.float32)
#         for k in range(no_of_lines):
#             if np.array_equal(topo[q], topo1[k]) and np.less_equal(np.absolute(eng[q] - eng1[k]), 2E-07).all():
#                 gen_num_eng = j
#                 pop_num_eng = k + 1
#                 break1 = 1
#                 break
#         if break1 == 1:
#             break
#     return gen_num_eng, pop_num_eng


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


def visualize(w, lx, ly, lz, penalty_coefficient, evaluation_version, max_rf22, parent_conn,
              file_io=True):
    topo_p, reslt_p = parent_import(w + 1, restart_pop=0)
    fitness_values = evaluation2(topo2=topo_p, reslt2=reslt_p, lx=lx, ly=ly, lz=lz,
                                 penalty_coefficient=penalty_coefficient, evaluation_version=evaluation_version,
                                 max_rf22=max_rf22)
    index = np.arange(topo_p.shape[0], dtype=int)
    pareto_front_index = pareto_front_finding(fitness_values, index)
    # index = reslt_p[pareto_front_index, :]  # correction: index >> reslt_p
    fitness_values_pareto = fitness_values[pareto_front_index]
    fv1_sort, fv2_sort, sort = fitval_sort(fitness_values_pareto)
    normalized_hypervolume, hypervolume = hypervolume_calculation(fv1_sort=fv1_sort, fv2_sort=fv2_sort, w=w,
                                                                  lx=lx, ly=ly, lz=lz,
                                                                  evaluation_version=evaluation_version,
                                                                  penalty_coefficient=penalty_coefficient,
                                                                  max_rf22=max_rf22)
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
    # ** pareto_front
    # fit_max, fit_min = max_fitval(w=w+1, lx=lx, ly=ly, lz=lz,
    #                               evaluation_version=evaluation_version, penalty_coefficient=penalty_coefficient,
    #                               max_rf22=max_rf22)
    # rp_max1 = fit_max[0] + 0.05 * (fit_max[0] - fit_min[0])
    # rp_min1 = fit_min[0] - 0.05 * (fit_max[0] - fit_min[0])
    # rp_max2 = fit_max[1] + 0.05 * (fit_max[1] - fit_min[1])
    # rp_min2 = fit_min[1] - 0.05 * (fit_max[1] - fit_min[1])
    if file_io:
        if os.path.isfile(f'Plot_data'):
            with open(f'Plot_data', mode='rb') as f_read:
                read_data = pickle.load(f_read)
            with open(f'Plot_data', mode='wb') as f_write:
                read_data.update({w: (fv1_sort, fv2_sort, w, normalized_hypervolume)})
                pickle.dump(read_data, f_write)
        else:
            with open(f'Plot_data', mode='wb') as f_write:
                pickle.dump({w: (fv1_sort, fv2_sort, w, normalized_hypervolume)}, f_write)
        parent_conn.send((fv1_sort, fv2_sort, w, normalized_hypervolume))
    print('Fitness Value 1,2 plot: ', fv1_sort, fv2_sort)
    # app.plot(0, fv1_sort, fv2_sort)
    # plt.figure(1)
    # plt.plot(fv1_sort, fv2_sort, marker='o', color='#2ca02c')
    # plt.xlabel('Objective function 1')
    # plt.ylabel('Objective function 2')
    # plt.axis((rp_min1, rp_max1, rp_min2, rp_max2))
    # ** hypervolume
    print('Hypervolume scatter: ', w, normalized_hypervolume)
    # app.scatter(1, w, normalized_hypervolume)
    # print(normalized_hypervolume, hypervolume)
    # plt.figure(2)
    # plt.scatter(w, normalized_hypervolume)
    # plt.xlabel('Iteration')
    # plt.ylabel('Hypervolume')
    # plt.axis((0, end_gen + 1, 0, 1))  # plt.axis((xmin, xmax, ymin, ymax))
    # plt.pause(0.1)
    print(sort)


def find_pareto_front_points(costs, return_index=False):
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


def visualize2(w, lx, ly, lz, penalty_coefficient, evaluation_version, max_rf22, parent_conn=None, file_io=True):
    topo_p, result_p = parent_import(w + 1, restart_pop=0)
    fitness_values = evaluation2(topo2=topo_p, reslt2=result_p, lx=lx, ly=ly, lz=lz,
                                 penalty_coefficient=penalty_coefficient, evaluation_version=evaluation_version,
                                 max_rf22=max_rf22)
    fitness_pareto = find_pareto_front_points(costs=fitness_values, return_index=False)
    fv1_pareto_sorted = fitness_pareto[:, 0]
    fv2_pareto_sorted = fitness_pareto[:, 1]
    hyper_volume = hypervolume_calculation2(fv1_sort=fv1_pareto_sorted, fv2_sort=fv2_pareto_sorted)
    if file_io:
        if os.path.isfile(f'Plot_data'):
            with open(f'Plot_data', mode='rb') as f_read:
                read_data = pickle.load(f_read)
            with open(f'Plot_data', mode='wb') as f_write:
                read_data.update({w: (fv1_pareto_sorted, fv2_pareto_sorted, w, hyper_volume)})
                pickle.dump(read_data, f_write)
        else:
            with open(f'Plot_data', mode='wb') as f_write:
                pickle.dump({w: (fv1_pareto_sorted, fv2_pareto_sorted, w, hyper_volume)}, f_write)
        parent_conn.send((fv1_pareto_sorted, fv2_pareto_sorted, w, hyper_volume))

    print('[VISUALIZE] Plotting Pareto front of fitness values:')
    print(f'> Objective function 1:{fv1_pareto_sorted}')
    print(f'> Objective function 2:{fv2_pareto_sorted}')
    print('[VISUALIZE] Scattering Hyper volume:')
    print(f'> Generation {w}: {hyper_volume}')
    return fv1_pareto_sorted, fv2_pareto_sorted, w, hyper_volume


def visualize3(w, lx, ly, lz, penalty_coefficient, evaluation_version, max_rf22, is_realtime, ref_x=0.0, ref_y=0.0,
               parent_conn=None, file_io=True):
    topo_p, result_p = parent_import(w + 1, restart_pop=0)
    fitness_values = evaluation2(topo2=topo_p, reslt2=result_p, lx=lx, ly=ly, lz=lz,
                                 penalty_coefficient=penalty_coefficient, evaluation_version=evaluation_version,
                                 max_rf22=max_rf22)
    if not is_realtime:
        fit_max, _ = max_fitval(w=w + 1, lx=lx, ly=ly, lz=ly, evaluation_version=evaluation_version,
                                penalty_coefficient=penalty_coefficient, max_rf22=max_rf22)
        ref_x = float(fit_max[0])
        ref_y = float(fit_max[1])
    fitness_pareto = find_pareto_front_points(costs=fitness_values, return_index=False)
    fv1_pareto_sorted = fitness_pareto[:, 0]
    fv2_pareto_sorted = fitness_pareto[:, 1]
    normalized_hv = hypervolume_calculation3(fv1_sort=fv1_pareto_sorted, fv2_sort=fv2_pareto_sorted,
                                             ref_x=ref_x, ref_y=ref_y)
    if file_io:
        if os.path.isfile(f'Plot_data'):
            with open(f'Plot_data', mode='rb') as f_read:
                read_data = pickle.load(f_read)
            with open(f'Plot_data', mode='wb') as f_write:
                read_data.update({w: (fv1_pareto_sorted, fv2_pareto_sorted, w, normalized_hv)})
                pickle.dump(read_data, f_write)
        else:
            with open(f'Plot_data', mode='wb') as f_write:
                pickle.dump({w: (fv1_pareto_sorted, fv2_pareto_sorted, w, normalized_hv)}, f_write)
        parent_conn.send((fv1_pareto_sorted, fv2_pareto_sorted, w, normalized_hv))

    print('[VISUALIZE] Plotting Pareto front of fitness values:')
    print(f'> Objective function 1:{fv1_pareto_sorted}')
    print(f'> Objective function 2:{fv2_pareto_sorted}')
    print('[VISUALIZE] Scattering Hyper volume:')
    print(f'> Generation {w}: {normalized_hv}')
    return fv1_pareto_sorted, fv2_pareto_sorted, w, normalized_hv


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


def selection(pop, fitness_values, pop_size):
    pop_index_0 = np.arange(pop.shape[0])
    pop_index = np.arange(pop.shape[0])
    pareto_front_index = []

    while len(pareto_front_index) < pop_size:
        new_pareto_front = pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
        total_pareto_size = len(pareto_front_index) + len(new_pareto_front)

        # check the size of pareto front, if larger than pop_size, remove some solutions using crowding criterion
        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(pareto_front_index)
            selected_solutions = (remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed))
            new_pareto_front = new_pareto_front[selected_solutions]

        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front))  # add to pareto
        # print('Total pareto front: ', pareto_front_index)
        remaining_index = set(pop_index) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))

    selected_pop_index = pareto_front_index.astype(int)
    selected_pop_topo = pop[pareto_front_index.astype(int)]
    return selected_pop_topo, selected_pop_index


def selection2(pop, fitness_values, pop_size):  # More efficient algorithm, 20 times faster than selection
    total_pop_idx = set(np.arange(len(pop)))
    remaining_population_idx = np.arange(len(pop))
    total_pareto_front_idx = set()

    while len(total_pareto_front_idx) < pop_size:
        new_pareto_front_idx = find_pareto_front_points(costs=fitness_values[remaining_population_idx],
                                                        return_index=True)
        total_pareto_size = len(total_pareto_front_idx) + len(new_pareto_front_idx)

        # check the size of pareto front, if larger than pop_size, remove some solutions using crowding criterion
        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(total_pareto_front_idx)
            selected_solutions = remove_using_crowding(fitness_values[new_pareto_front_idx], number_solutions_needed)
            new_pareto_front_idx = new_pareto_front_idx[selected_solutions]

        total_pareto_front_idx = total_pareto_front_idx.union(remaining_population_idx[new_pareto_front_idx])
        remaining_population_idx = np.array(list(total_pop_idx - total_pareto_front_idx))
    selected_pop_topo = pop[list(total_pareto_front_idx)]
    return selected_pop_topo, list(total_pareto_front_idx)


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
    from GraphicUserInterface import plot_test
    from time import sleep

    path = r'F:\shshsh\data-23-1-4'
    number_of_generations = 19

    os.chdir(path)
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=((1400 - 5) / 100, (700 - 5) / 100), dpi=100)
    axes[0].set(title='Pareto Fronts', xlabel='Objective function 1', ylabel='Objective function 2')
    axes[1].set(title='Hyper Volume by Generation', xlabel='Generation', ylabel='Hyper volume')
    axes[0].grid(True)
    axes[1].grid(True)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    for gen_idx in range(number_of_generations):
        px, py, sx, sy = visualize3(w=gen_idx + 1, lx=10, ly=10, lz=10, penalty_coefficient=0.1,
                                    evaluation_version='ver3', is_realtime=True, ref_x=0.2, ref_y=1.75,
                                    max_rf22=9900.0, parent_conn=None, file_io=False)
        plot_test(axes, px, py, sx, sy)
    plt.show()
