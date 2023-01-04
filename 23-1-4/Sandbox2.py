import pickle
import os
from time import sleep
import numpy as np
import timeit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline


def test1():
    arr = np.arange(81).reshape((3, 3, 3, 3))
    arr2 = np.swapaxes(arr, axis1=1, axis2=3)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for w in range(3):
                    print(arr[w, i, j, k] == arr2[w, k, j, i])


def test2_1():
    age, height = 16, 1.24


def test2_2():
    age = 16
    height = 1.24


def test2():
    print(timeit.timeit(test2_1, number=10000000))
    print(timeit.timeit(test2_2, number=10000000))


def test3_1():
    arr = np.arange(81).reshape((3, 3, 3, 3))
    s = len(arr)


def test3_2():
    mrr = np.arange(81).reshape((3, 3, 3, 3))
    m = mrr.shape[0]


def test3():
    print(timeit.timeit(test3_1, number=100000))
    print(timeit.timeit(test3_2, number=100000))


def plot_previous_data():
    if os.path.isfile(f'Plot_data'):
        with open(f'Plot_data', mode='rb') as f_read:
            read_data = pickle.load(f_read)
        for key, value in read_data.items():
            print(key, value[0].shape, value[1].shape, value[2], value[3])


def make_spline(x, y, ax):
    print(x, y)
    xy = np.vstack((x, y))
    xy = xy[:, np.argsort(xy[0, :])]
    xy_spline = make_interp_spline(xy[0], xy[1], k=3)

    print('lenx: ', len(x))
    x_new = np.linspace(x.min(), x.max(), 100)
    y_new = xy_spline(x_new)
    return x_new, y_new


def plot_test(px, py, sx, sy):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=100)
    ax[0].set(title='Objective Functions', xlabel='Objective function 1', ylabel='Objective function 2')
    ax[1].set(title='Hyper Volume by Generation', xlabel='Generation', ylabel='Hyper volume')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].plot(px, py, marker='o')
    ax[1].scatter(sx, sy, marker='o')
    plt.show()


def find_pareto_front_points(costs, return_index=False):
    # costs: (no_of_points x no_of_costs) array
    # return: (no_of_costs x no_of_pareto_points) array
    print(costs.shape)
    costs_copy = costs.copy()
    unsorted_indices = np.arange(costs_copy.shape[0])
    next_point_idx = 0
    while next_point_idx < len(costs_copy):
        not_dominated_point_mask = np.any(costs_copy < costs_copy[next_point_idx], axis=1)
        not_dominated_point_mask[next_point_idx] = True
        unsorted_indices = unsorted_indices[not_dominated_point_mask]  # Remove dominated points
        costs_copy = costs_copy[not_dominated_point_mask]
        next_point_idx = np.sum(not_dominated_point_mask[:next_point_idx]) + 1
    # pareto_front_unsorted = costs[:, unsorted_indices]
    sorted_indices = unsorted_indices[np.argsort(costs[:, unsorted_indices][0, :])]
    pareto_front_sorted = costs[:, sorted_indices]
    print('Index: ', sorted_indices)
    if return_index:
        return sorted_indices
    else:
        return pareto_front_sorted


def find_pareto_front_points(costs, return_index=False):
    # costs: (no_of_points x no_of_costs) array
    # return: (no_of_costs x no_of_pareto_points) array
    print(costs.shape)
    costs_copy = costs.copy()
    unsorted_indices = np.arange(costs_copy.shape[0])
    next_point_idx = 0
    while next_point_idx < len(costs_copy):
        not_dominated_point_mask = np.any(costs_copy < costs_copy[next_point_idx], axis=1)
        not_dominated_point_mask[next_point_idx] = True
        unsorted_indices = unsorted_indices[not_dominated_point_mask]  # Remove dominated points
        costs_copy = costs_copy[not_dominated_point_mask]
        next_point_idx = np.sum(not_dominated_point_mask[:next_point_idx]) + 1
    # pareto_front_unsorted = costs[:, unsorted_indices]
    sorted_indices = unsorted_indices[np.argsort(costs[unsorted_indices][:, 0])]
    pareto_front_sorted = costs[sorted_indices]
    print(sorted_indices)
    if return_index:
        return sorted_indices
    else:
        return pareto_front_sorted


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


if __name__ == '__main__':
    path = r'F:\shshsh\data-23-1-4'
    os.chdir(path)
    # with open('Plot_data', 'rb') as f:
    #     px, py, sx, sy = pickle.load(f)[1]

    from PostProcessing import visualize2

    px, py, sx, sy = visualize2(w=1, lx=10, ly=10, lz=10, penalty_coefficient=0.1, evaluation_version='ver3',
                                max_rf22=9900.0, parent_conn=None, file_io=False)
    plot_test(px, py, sx, sy)


