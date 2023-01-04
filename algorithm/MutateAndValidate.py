import datetime as dt
from scipy.ndimage import label
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import random
import timeit


@njit
def quaver_to_full(quaver):
    quarter = np.concatenate((np.flip(quaver, axis=0), quaver), axis=0)
    half = np.concatenate((np.flip(quarter, axis=1), quarter), axis=1)
    full = np.concatenate((np.flip(half, axis=2), half), axis=2)
    return np.swapaxes(full, axis1=0, axis2=2)


def visualize_one_cube(cube_3d_array, full=False):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={
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


@njit
def design_const_add(topologies, lx, ly, lz):
    topo = topologies.copy()
    flag = 1
    direc = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

    while flag:
        # a=shuffle range(a)
        flag = 0
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    topcrs = 0
                    if topo[k][j][i] == 1:
                        for m in range(4):
                            if k + direc[m][0] in range(lz) and j + direc[m][1] in range(ly):
                                if topo[k + direc[m][0]][j + direc[m][1]][i] == 1:
                                    if topo[k + direc[m][0]][j][i] == 0 and topo[k][j + direc[m][1]][i] == 0:
                                        randbit = random.getrandbits(1)
                                        if randbit == 1:
                                            topo[k + direc[m][0]][j][i] = 1
                                        else:
                                            topo[k][j + direc[m][1]][i] = 1
                                        topcrs = 1
                                        flag = 1
                                        break

                            if k + direc[m][0] in range(lz) and i + direc[m][1] in range(lx):
                                if topo[k + direc[m][0]][j][i + direc[m][1]] == 1:
                                    if topo[k + direc[m][0]][j][i] == 0 and topo[k][j][i + direc[m][1]] == 0:
                                        randbit = random.getrandbits(1)
                                        if randbit == 1:
                                            topo[k + direc[m][0]][j][i] = 1
                                        else:
                                            topo[k][j][i + direc[m][1]] = 1
                                        topcrs = 1
                                        flag = 1
                                        break

                            if j + direc[m][0] in range(ly) and i + direc[m][1] in range(lx):
                                if topo[k][j + direc[m][0]][i + direc[m][1]] == 1:
                                    if topo[k][j + direc[m][0]][i] == 0 and topo[k][j][i + direc[m][1]] == 0:
                                        randbit = random.getrandbits(1)
                                        if randbit == 1:
                                            topo[k][j + direc[m][0]][i] = 1
                                        else:
                                            topo[k][j][i + direc[m][1]] = 1
                                        topcrs = 1
                                        flag = 1
                                        break

                            for z in [1, -1]:
                                if k + direc[m][0] in range(lz) and j + direc[m][1] in range(ly) and i + z in range(lx):
                                    if topo[k + direc[m][0]][j + direc[m][1]][i + z] == 1:
                                        if topo[k + direc[m][0]][j][i] == 0 and topo[k][j + direc[m][1]][i] == 0 and \
                                                topo[k][j][i + z] == 0:
                                            if topo[k + direc[m][0]][j + direc[m][1]][i] == 0 and \
                                                    topo[k + direc[m][0]][j][i + z] == 0 and topo[k][j + direc[m][1]][
                                                i + z] == 0:
                                                rand1 = random.randint(1, 3)
                                                rand2 = random.randint(1, 2)

                                                if rand1 == 1:
                                                    topo[k + direc[m][0]][j][i] = 1
                                                    if rand2 == 1:
                                                        topo[k + direc[m][0]][j + direc[m][1]][i] = 1
                                                    else:
                                                        topo[k + direc[m][0]][j][i + z] = 1

                                                if rand1 == 2:
                                                    topo[k][j + direc[m][1]][i] = 1
                                                    if rand2 == 1:
                                                        topo[k + direc[m][0]][j + direc[m][1]][i] = 1

                                                    else:
                                                        topo[k][j + direc[m][1]][i + z] = 1

                                                if rand1 == 3:
                                                    topo[k][j][i + z] = 1
                                                    if rand2 == 1:
                                                        topo[k + direc[m][0]][j][i + z] = 1
                                                    else:
                                                        topo[k][j + direc[m][1]][i + z] = 1

                                                topcrs = 1
                                                flag = 1
    is_no_change = np.array_equal(topologies, topo)
    # print('is no change?', is_no_change)
    return topo, is_no_change


def check_printability_by_slicing3(arr_3d, max_distance=1):  # arr[x,y,z]
    arr_3d_result = arr_3d.copy()
    x_size, y_size, z_size = arr_3d_result.shape[0], arr_3d_result.shape[1], arr_3d_result.shape[2]
    total_changed_voxels = 0
    y_search_range = range(1, y_size - 1)
    y_search_range_reversed = range(y_size - 2, 0, -1)
    is_no_change = False

    while True:
        changed_voxels = 0
        for y_direction in (1, -1):
            for y_idx in y_search_range if y_direction == 1 else y_search_range_reversed:
                # Declaration of arr_2d_quarter and labeled_arr
                labeled_arr, max_island_idx = label(arr_3d_result[:, y_idx, :])
                dead_islands, survived_islands = dead_and_survived_islands(y_idx=y_idx, y_direction=y_direction,
                                                                           x_size=x_size, z_size=z_size,
                                                                           max_island_idx=max_island_idx,
                                                                           labeled_arr=labeled_arr,
                                                                           arr_3d_result=arr_3d_result)
                # Eliminating bad voxels by dead islands and survived islands with bad angle
                arr_3d_result, changed_voxels = voxel_elimination_by_islands(
                    x_size=x_size, z_size=z_size, labeled_arr=labeled_arr, dead_islands=np.array(list(dead_islands)),
                    survived_islands=np.array(list(survived_islands)), arr_3d_result=arr_3d_result, y_idx=y_idx,
                    max_distance=max_distance, changed_voxels=changed_voxels, y_direction=y_direction)
        total_changed_voxels += changed_voxels
        # print('Changed voxels: ', changed_voxels)
        if changed_voxels == 0:
            break
    # print('Eliminated voxels in printability: ', total_changed_voxels)
    if np.array_equal(arr_3d, arr_3d_result):
        is_no_change = True
    return arr_3d_result, is_no_change


@njit
def dead_and_survived_islands(y_idx, y_direction, x_size, z_size, max_island_idx, labeled_arr, arr_3d_result):
    survived_islands, dead_islands = set(), set()
    # Determining which ones are the dead islands
    for x_idx in range(x_size):
        for z_idx in range(z_size):
            island_idx = labeled_arr[x_idx, z_idx]
            if island_idx and arr_3d_result[x_idx, y_idx - y_direction, z_idx]:
                survived_islands.add(island_idx)
    for i in range(1, max_island_idx + 1):
        if i not in survived_islands:
            dead_islands.add(i)
    return dead_islands, survived_islands


@njit
def voxel_elimination_by_islands(x_size, z_size, labeled_arr, dead_islands, survived_islands, arr_3d_result, y_idx,
                                 max_distance, changed_voxels, y_direction):
    for x_idx in range(x_size):
        for z_idx in range(z_size):
            island_idx = labeled_arr[x_idx, z_idx]
            if island_idx in dead_islands:
                arr_3d_result[x_idx, y_idx, z_idx] = 0
                changed_voxels += 1

            if island_idx in survived_islands:
                is_any_around = False
                for x_distance in range(-max_distance, max_distance + 1):
                    for z_distance in range(-max_distance, max_distance + 1):
                        x_position = x_idx + x_distance
                        z_position = z_idx + z_distance
                        if x_position < 0:
                            x_position = 0
                        elif x_position > x_size - 1:
                            x_position = x_size - 1
                        if z_position < 0:
                            z_position = 0
                        elif z_position > x_size - 1:
                            z_position = x_size - 1
                        if arr_3d_result[x_position, y_idx - y_direction, z_position]:
                            is_any_around = True
                if not is_any_around:
                    # print(f'Overhang island Eliminated (x,y,z) = ({x_idx},{y_idx},{z_idx})')
                    arr_3d_result[x_idx, y_idx, z_idx] = 0
                    changed_voxels += 1
    return arr_3d_result, changed_voxels


def one_connected_tree(arr_3d, add_probability):
    arr_shape = arr_3d.shape
    arr_copy = arr_3d.copy()
    is_no_change = False
    while True:
        # total_changed_voxels = 0
        labeled_arr, max_label_idx = label(arr_copy)
        survived_labels = set.intersection(
            *one_survived_tree_labels(labeled_arr, arr_shape[0], arr_shape[1], arr_shape[2])) - {0}
        # print('Survived labels: ', survived_labels)

        unique = dict(zip(*np.unique(labeled_arr, return_counts=True)))
        unique = {key: unique[key] for key in survived_labels}
        try:
            last_survived_label = max(unique, key=unique.get)
            arr_copy = one_survived_tree(arr_copy, labeled_arr, arr_shape[0], arr_shape[1],
                                         arr_shape[2], last_survived_label)
            # print('last survived label: ', last_survived_label, unique[last_survived_label])
            # print('Eliminated voxels in cut tree: ', total_changed_voxels)
            break
        except ValueError:
            arr_copy = add_random_voxels(arr_copy, probability=add_probability)
    if np.array_equal(arr_3d, arr_copy):
        is_no_change = True
    return arr_copy, is_no_change


@njit
def one_survived_tree_labels(labeled_arr, lx, ly, lz):
    labels_plane_1 = set()
    labels_plane_2 = set()
    labels_plane_3 = set()
    labels_plane_4 = set()
    labels_plane_5 = set()
    labels_plane_6 = set()

    for i in range(ly):
        for j in range(lz):
            labels_plane_1.add(labeled_arr[0, i, j])
    for i in range(ly):
        for j in range(lz):
            labels_plane_2.add(labeled_arr[lx - 1, i, j])
    for i in range(lx):
        for j in range(lz):
            labels_plane_3.add(labeled_arr[i, 0, j])
    for i in range(lx):
        for j in range(lz):
            labels_plane_4.add(labeled_arr[i, ly - 1, j])
    for i in range(lx):
        for j in range(ly):
            labels_plane_5.add(labeled_arr[i, j, 0])
    for i in range(lx):
        for j in range(ly):
            labels_plane_6.add(labeled_arr[i, j, lz - 1])
    return labels_plane_1, labels_plane_2, labels_plane_3, labels_plane_4, labels_plane_5, labels_plane_6


@njit
def one_survived_tree(arr_copy, labeled_arr, lx, ly, lz, last_survived_label):
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                if labeled_arr[i, j, k] not in (0, last_survived_label):
                    arr_copy[i, j, k] = 0
    return arr_copy


@njit
def connect_island(local_arr, labeled_arr, max_label, x_start_gap, y_start_gap, z_start_gap):
    arr_copy = local_arr.copy()
    llx = local_arr.shape[0]
    lly = local_arr.shape[1]
    llz = local_arr.shape[2]
    changed_voxels = 0
    main_label = labeled_arr[1 - x_start_gap, 1 - y_start_gap, 1 - z_start_gap]
    other_labels = set([label_idx for label_idx in range(1, max_label + 1)])
    other_labels.remove(main_label)
    for i in range(llx):
        for j in range(lly):
            for k in range(llz):
                # print('ijk: ', i, j, k, 'start gap: ', x_start_gap, y_start_gap, z_start_gap, 'end gap', x_end_gap, y_end_gap, z_end_gap)
                if labeled_arr[i, j, k] in other_labels:
                    changed_voxels += 1
                    arr_copy[i, j, k] = 0
        return arr_copy, changed_voxels


def validate_mutated_topology(arr_3d, mutation_probability, add_probability, timeout):
    arr_3d_mutated = mutation(arr_3d.copy(), mutation_probability=mutation_probability)
    lx = arr_3d.shape[0]
    ly = arr_3d.shape[1]
    lz = arr_3d.shape[2]
    while True:
        arr_3d_mutated_copy = arr_3d_mutated.copy()
        now1 = dt.datetime.now()
        is_timeout = False
        while not is_timeout:
            now2 = dt.datetime.now()
            time_diff = now2 - now1
            if time_diff.seconds + time_diff.microseconds * 1e-6 >= timeout:
                is_timeout = True
            arr_3d_mutated_copy, not_changed1 = one_connected_tree(arr_3d_mutated_copy, add_probability=add_probability)
            arr_3d_mutated_copy, not_changed2 = check_printability_by_slicing3(arr_3d_mutated_copy)
            arr_3d_mutated_copy, not_changed3 = design_const_add(arr_3d_mutated_copy, lx, ly, lz)
            if not_changed1 and not_changed2 and not_changed3:  # while 문 처음과 끝에서 변한 구조가 없을 때 break
                break
        if not is_timeout:
            break
        # print('Timeout!')
        arr_3d_mutated = mutation(arr_3d, mutation_probability=0.1)
        # visualize_one_cube(arr_copy)
    return arr_3d_mutated_copy


def validate_mutated_topologies(arr_4d, mutation_probability, add_probability, timeout, view_topo=False):
    arr_4d_copy = arr_4d.copy()
    for arr_idx, arr_3d in enumerate(arr_4d):
        # print(arr_idx)
        arr_4d_copy[arr_idx] = validate_mutated_topology(arr_3d, mutation_probability=mutation_probability,
                                                         timeout=timeout, add_probability=add_probability)
        if view_topo:
            visualize_one_cube(arr_4d_copy[arr_idx])
    return arr_4d_copy


@njit
def mutation(arr_3d, mutation_probability):
    arr_copy = arr_3d.copy()
    lx = arr_3d.shape[0]
    ly = arr_3d.shape[1]
    lz = arr_3d.shape[2]

    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                p = np.random.random()
                if p < mutation_probability:
                    arr_copy[i, j, k] = 1
                else:
                    arr_copy[i, j, k] = 0
    return arr_copy


@njit
def move_one_random_voxel(arr_3d, probability=0.01):
    arr_copy = arr_3d.copy()
    lx = arr_3d.shape[0]
    ly = arr_3d.shape[1]
    lz = arr_3d.shape[2]

    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                if arr_copy[i, j, k] and (np.random.random() < probability):
                    arr_copy[i, j, k] = 0
                    p = np.random.random()
                    r1, r2, r3 = 0, 0, 0
                    if p < 1 / 6:
                        r1 = -1
                    elif 1 / 6 < p < 1 / 3:
                        r1 = 1
                    elif 1 / 3 < p < 1 / 2:
                        r2 = -1
                    elif 1 / 2 < p < 2 / 3:
                        r2 = 1
                    elif 2 / 3 < p < 5 / 6:
                        r3 = -1
                    else:
                        r3 = 1
                    if i + r1 < 0:
                        r1 = 1

                    arr_copy[i + r1, j + r2, k + r3] = 1


@njit
def add_random_voxels(arr_3d, probability=0.01):
    arr_copy = arr_3d.copy()
    lx = arr_3d.shape[0]
    ly = arr_3d.shape[1]
    lz = arr_3d.shape[2]

    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                if (not arr_copy[i, j, k]) and (np.random.random() < probability):
                    arr_copy[i, j, k] = 1
    return arr_copy


if __name__ == '__main__':
    path = rf'C:\Users\dcas\PythonCodes\Coop\pythoncode\10x10x10 - 복사본\topo_parent_1.csv'
    cube_4d_array = np.genfromtxt(path, dtype=int, delimiter=',').reshape((100, 10, 10, 10))
    t = timeit.Timer(lambda: validate_mutated_topologies(cube_4d_array, mutation_probability=0.05, add_probability=0.01,
                                                         timeout=0.5, view_topo=False))
    test_iteration = 10
    print('Begin Testing...')
    print(f'Total runtime of {test_iteration} generations: {t.timeit(test_iteration)}s')
    # 10 generation runtime: 39.84s on i5-8600K
