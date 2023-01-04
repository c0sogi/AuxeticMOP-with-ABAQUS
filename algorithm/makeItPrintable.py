from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt
import itertools
import timeit
from numba import njit
import random


def quaver_to_full(quaver):
    quarter = np.concatenate((np.flip(quaver, axis=0), quaver), axis=0)
    half = np.concatenate((np.flip(quarter, axis=1), quarter), axis=1)
    full = np.concatenate((np.flip(half, axis=2), half), axis=2)
    return np.swapaxes(full, axis1=0, axis2=2)


def visualize_one_cube(cube_3d_array, full=False):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20), subplot_kw={
        'projection': '3d'
    })
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.voxels(quaver_to_full(cube_3d_array) if full else cube_3d_array, facecolor='#e02050', edgecolors='k')
    ax.grid(True)
    plt.show()


def visualize_n_cubes(cube_4d_array, full=False):
    n_row = 1
    n_col = cube_4d_array.shape[0]
    fig, ax = plt.subplots(n_row, n_col, figsize=(20, 5), subplot_kw={
        'projection': '3d'
    })
    for idx in range(n_col):
        ax[idx].set(xlabel='x', ylabel='y', zlabel='z')
        ax[idx].voxels(quaver_to_full(cube_4d_array[idx]) if full else cube_4d_array[idx],
                       facecolor='r', edgecolors='k')
        ax[idx].grid(True)
    plt.show()


def check_printability_by_slicing(arr_3d):  # arr[x,y,z]
    arr_3d_result = arr_3d.copy()
    x_size, y_size, z_size = arr_3d_result.shape[0], arr_3d_result.shape[1], arr_3d_result.shape[2]

    for y_direction in (1, -1):
        for y_idx in range(1, y_size - 1) if y_direction == 1 else reversed(range(1, y_size - 1)):
            # Declaration of arr_2d_quarter and labeled_arr
            survived_islands = set()
            labeled_arr, max_island_idx = label(arr_3d_result[:, y_idx, :])

            # Determining which ones are the dead islands
            for x_idx in range(x_size):
                for z_idx in range(z_size):
                    island_idx = labeled_arr[x_idx, z_idx]
                    survived_islands.add(island_idx) if (
                            island_idx and arr_3d_result[x_idx, y_idx - y_direction, z_idx]) else None
            dead_islands = set([i for i in range(1, max_island_idx + 1)]) - survived_islands

            # Eliminating bad pixels by dead islands
            for x_idx in range(x_size):
                for z_idx in range(z_size):
                    island_idx = labeled_arr[x_idx, z_idx]
                    if island_idx in dead_islands:
                        arr_3d_result[x_idx, y_idx, z_idx] = 0
    return arr_3d_result


def check_printability_by_slicing2(arr_3d):  # arr[x,y,z]
    arr_3d_result = arr_3d.copy()
    x_size, y_size, z_size = arr_3d_result.shape[0], arr_3d_result.shape[1], arr_3d_result.shape[2]
    for y_direction in (1, -1):
        for y_idx in range(1, y_size - 1) if y_direction == 1 else reversed(range(1, y_size - 1)):
            # Initialization of set and arrays
            survived_islands = set()
            labeled_arr, max_island_idx = label(arr_3d_result[:, y_idx, :])

            # Determining which ones are the dead islands
            for x_idx, z_idx in itertools.product(range(x_size), range(z_size)):
                island_idx = labeled_arr[x_idx, z_idx]
                survived_islands.add(island_idx) if (
                            island_idx and arr_3d_result[x_idx, y_idx - y_direction, z_idx]) else None
            dead_islands = set([island_idx for island_idx in range(1, max_island_idx + 1)]) - survived_islands

            # Eliminating bad pixels by dead islands
            for x_idx, z_idx in itertools.product(range(x_size), range(z_size)):
                island_idx = labeled_arr[x_idx, z_idx]
                arr_3d_result[x_idx, y_idx, z_idx] = 0 if (
                            (not arr_3d_result[x_idx, y_idx, z_idx]) or (island_idx in dead_islands)) else 1
    return arr_3d_result


def check_printability_by_slicing3(arr_3d):  # arr[x,y,z]
    arr_3d_result = arr_3d.copy()
    x_size, y_size, z_size = arr_3d_result.shape[0], arr_3d_result.shape[1], arr_3d_result.shape[2]

    for y_direction in (1, -1):
        for y_idx in range(1, y_size - 1) if y_direction == 1 else reversed(range(1, y_size - 1)):
            # Declaration of arr_2d_quarter and labeled_arr
            survived_islands = set()
            labeled_arr, max_island_idx = label(arr_3d_result[:, y_idx, :])
            dead_islands = determine_dead_islands(y_idx=y_idx, y_direction=y_direction, x_size=x_size, z_size=z_size,
                                                  max_island_idx=max_island_idx, labeled_arr=labeled_arr,
                                                  arr_3d_result=arr_3d_result)
            # Eliminating bad pixels by dead islands
            for x_idx in range(x_size):
                for z_idx in range(z_size):
                    island_idx = labeled_arr[x_idx, z_idx]
                    if island_idx in dead_islands:
                        arr_3d_result[x_idx, y_idx, z_idx] = 0
    # visualize_one_cube(arr_3d_result)
    return arr_3d_result


@njit
def determine_dead_islands(y_idx, y_direction, x_size, z_size, max_island_idx, labeled_arr, arr_3d_result):
    survived_islands, dead_islands = set(), set()
    # Determining which ones are the dead islands
    for x_idx in range(x_size):
        for z_idx in range(z_size):
            island_idx = labeled_arr[x_idx, z_idx]
            if island_idx and arr_3d_result[x_idx, y_idx - y_direction, z_idx]:
                survived_islands.add(island_idx)
    for i in range(1, max_island_idx + 1):
        if not i in survived_islands:
            dead_islands.add(i)
    return dead_islands


def random_array(shape, probability):
    rd_arr = np.zeros((shape[0], shape[1], shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                rd_arr[i, j, k] = 1 if random.random() < probability else 0
    return rd_arr


if __name__ == '__main__':
    path = rf'E:\pythoncode\22-12-28\data - original\topo_offspring_1.csv'
    cube_4d_array = np.genfromtxt(path, dtype=int, delimiter=',').reshape((100, 10, 10, 10))
    f1 = lambda: [check_printability_by_slicing(cube_4d_array[i]) for i in range(100)]  # 380ms
    f2 = lambda: [check_printability_by_slicing2(cube_4d_array[i]) for i in range(100)]  # 344ms
    f3 = lambda: [check_printability_by_slicing3(cube_4d_array[i]) for i in range(100)]  # 344ms
    # print(timeit.timeit(f1, number=100))  # 13.46s
    # print(timeit.timeit(f2, number=100))  # 17.27s
    print(timeit.timeit(f3, number=100))  # 10.49s
    # rd_arr = random_array(shape=(10, 10, 10), probability=0.35)
    # print('rd arr created')
    # check_printability_by_slicing3(rd_arr)