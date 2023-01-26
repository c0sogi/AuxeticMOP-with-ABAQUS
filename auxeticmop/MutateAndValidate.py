# import datetime as dt
from scipy.ndimage import label
from numba import njit
import numpy as np
import random


def make_3d_print_without_support(arr_3d: np.ndarray, max_distance: int = 1) -> tuple[np.ndarray, bool, int]:
    arr_3d_result = arr_3d.copy()
    x_size, y_size, z_size = arr_3d_result.shape
    total_changed_voxels = 0
    y_search_range = range(1, y_size)  # 1~9 / original: range(1, y_size - 1)
    y_search_range_reversed = range(y_size - 2, -1, -1)  # 8~0 / original: range(y_size - 2, 1, -1)
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
        if changed_voxels == 0:
            break
    if np.array_equal(arr_3d, arr_3d_result):
        is_no_change = True
    return arr_3d_result, is_no_change, total_changed_voxels


@njit
def dead_and_survived_islands(y_idx: int, y_direction: int, x_size: int, z_size: int, max_island_idx: int,
                              labeled_arr: np.ndarray, arr_3d_result: np.ndarray) -> tuple[set, set]:
    survived_islands = set()
    dead_islands = set()
    # Determining which ones are the dead islands
    for x_idx in range(x_size):
        for z_idx in range(z_size):
            island_label_num = labeled_arr[x_idx, z_idx]
            if island_label_num and arr_3d_result[x_idx, y_idx - y_direction, z_idx]:  # y_idx = 1~9 -> 0~8
                survived_islands.add(island_label_num)
    for island_label_num in range(1, max_island_idx + 1):
        if island_label_num not in survived_islands:
            dead_islands.add(island_label_num)
    return dead_islands, survived_islands


@njit
def voxel_elimination_by_islands(x_size: int, z_size: int, labeled_arr: np.ndarray, dead_islands: np.ndarray,
                                 survived_islands: np.ndarray, arr_3d_result: np.ndarray, y_idx: int,
                                 max_distance: int, changed_voxels: int, y_direction: int) -> tuple[np.ndarray, int]:
    for x_idx in range(x_size):
        for z_idx in range(z_size):
            island_idx = labeled_arr[x_idx, z_idx]
            if island_idx in dead_islands:
                arr_3d_result[x_idx, y_idx, z_idx] = 0
                changed_voxels -= 1

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
                        elif z_position > z_size - 1:
                            z_position = z_size - 1
                        if arr_3d_result[x_position, y_idx - y_direction, z_position]:
                            is_any_around = True
                if not is_any_around:
                    # print(f'Overhang island Eliminated (x,y,z) = ({x_idx},{y_idx},{z_idx})')
                    arr_3d_result[x_idx, y_idx, z_idx] = 0
                    changed_voxels -= 1
    return arr_3d_result, changed_voxels


def one_connected_tree(arr_3d: np.ndarray) -> tuple[np.ndarray | None, bool, int]:
    arr_shape = arr_3d.shape
    arr_copy = arr_3d.copy()
    total_changed_voxels = 0
    while True:
        labeled_arr, max_label_idx = label(arr_copy)
        survived_labels = set.intersection(
            *one_survived_tree_labels(labeled_arr, arr_shape[0], arr_shape[1], arr_shape[2])) - {0}
        unique = dict(zip(*np.unique(labeled_arr, return_counts=True)))
        unique = {key: unique[key] for key in survived_labels}
        try:
            last_survived_label = max(unique, key=unique.get)
            arr_copy, changed_voxels = one_survived_tree(
                arr_3d=arr_copy, labeled_arr=labeled_arr, last_survived_label=last_survived_label)
            total_changed_voxels += changed_voxels
            break
        except ValueError:
            return None, False, total_changed_voxels
    return arr_copy, np.array_equal(arr_3d, arr_copy), total_changed_voxels


@njit
def one_survived_tree_labels(labeled_arr: np.ndarray, lx: int, ly: int, lz: int) -> tuple[set, set, set, set, set, set]:
    labels_plane_1 = set()
    labels_plane_2 = set()
    labels_plane_3 = set()
    labels_plane_4 = set()
    labels_plane_5 = set()
    labels_plane_6 = set()

    for idx_1 in range(ly):
        for idx_2 in range(lz):
            labels_plane_1.add(labeled_arr[0, idx_1, idx_2])
    for idx_1 in range(ly):
        for idx_2 in range(lz):
            labels_plane_2.add(labeled_arr[lx - 1, idx_1, idx_2])
    for idx_1 in range(lx):
        for idx_2 in range(lz):
            labels_plane_3.add(labeled_arr[idx_1, 0, idx_2])
    for idx_1 in range(lx):
        for idx_2 in range(lz):
            labels_plane_4.add(labeled_arr[idx_1, ly - 1, idx_2])
    for idx_1 in range(lx):
        for idx_2 in range(ly):
            labels_plane_5.add(labeled_arr[idx_1, idx_2, 0])
    for idx_1 in range(lx):
        for idx_2 in range(ly):
            labels_plane_6.add(labeled_arr[idx_1, idx_2, lz - 1])
    return labels_plane_1, labels_plane_2, labels_plane_3, labels_plane_4, labels_plane_5, labels_plane_6


@njit
def one_survived_tree(arr_3d: np.ndarray, labeled_arr: np.ndarray, last_survived_label: int) -> tuple[np.ndarray, int]:
    lx = arr_3d.shape[0]
    ly = arr_3d.shape[1]
    lz = arr_3d.shape[2]
    changed_voxels = 0
    for idx_1 in range(lx):
        for idx_2 in range(ly):
            for idx_3 in range(lz):
                if labeled_arr[idx_1, idx_2, idx_3] not in (0, last_survived_label):
                    changed_voxels -= 1
                    arr_3d[idx_1, idx_2, idx_3] = 0
    return arr_3d, changed_voxels


def mutate_and_validate_topology(arr_3d: np.ndarray, mutation_probability: float) -> None | np.ndarray:
    # timeout_seconds = dt.timedelta(seconds=timeout)
    arr_3d_mutated, total_voxels_0 = mutation(arr_3d.copy(), mutation_probability=mutation_probability)
    lx, ly, lz = arr_3d.shape
    while True:
        arr_3d_mutated_copy = arr_3d_mutated.copy()
        total_voxels_1, total_voxels_2, total_voxels_3 = 0, 0, 0
        is_timeout = False
        while not is_timeout:
            _arr_3d_mutated_copy = arr_3d_mutated_copy.copy()  #
            arr_3d_mutated_copy, not_changed1, voxels_1 = one_connected_tree(arr_3d_mutated_copy)
            if arr_3d_mutated_copy is None:
                return None
            arr_3d_mutated_copy, not_changed2, voxels_2 = make_3d_print_without_support(arr_3d_mutated_copy)
            arr_3d_mutated_copy, not_changed3, voxels_3 = make_voxels_surface_contact(arr_3d_mutated_copy, lx, ly, lz)
            total_voxels_1 += voxels_1
            total_voxels_2 += voxels_2
            total_voxels_3 += voxels_3
            if not_changed1 and not_changed2 and not_changed3:  # while 문 처음과 끝에서 변한 구조가 없을 때 break
                break
            if np.array_equal(arr_3d_mutated_copy, _arr_3d_mutated_copy):  #
                is_timeout = True
                break
            # if abs(total_voxels_0 + total_voxels_1 + total_voxels_2 + total_voxels_3) > mutation_probability * lx * ly * lz:
            #     is_timeout = True  # Timeout due to too much voxel changes in structure
            #     break
            # now2 = dt.datetime.now()
            # time_diff = now2 - now1
            # if time_diff > timeout_seconds:
            #     is_timeout = True  # Timeout due to too much long validation time
            #     break
        if is_timeout:
            arr_3d_mutated, total_voxels_0 = mutation(arr_3d, mutation_probability=mutation_probability)
        else:
            break
        # Timeout!
    total_voxels_change = total_voxels_0 + total_voxels_1 + total_voxels_2 + total_voxels_3
    print('> Total changes in structure during validation < ')
    print(f"- Voxel number changes: [{total_voxels_change}] voxels", end=' | ')
    print(f"Changed of volume fraction: [{100 * total_voxels_change/ (lx*ly*lz):.2f}] %")
    return arr_3d_mutated_copy


@njit
def mutation(arr_3d: np.ndarray, mutation_probability: float) -> tuple[np.ndarray, int]:
    arr_copy = arr_3d.copy()
    lx = arr_3d.shape[0]
    ly = arr_3d.shape[1]
    lz = arr_3d.shape[2]
    changed_voxels = 0

    for idx_1 in range(lx):
        for idx_2 in range(ly):
            for idx_3 in range(lz):
                p = np.random.random()
                if p < mutation_probability:
                    if arr_copy[idx_1, idx_2, idx_3] == 1:
                        arr_copy[idx_1, idx_2, idx_3] = 0
                        changed_voxels -= 1
                    else:
                        arr_copy[idx_1, idx_2, idx_3] = 1
                        changed_voxels += 1
    return arr_copy, changed_voxels


@njit
def make_voxels_surface_contact(topologies: np.ndarray, lx: int, ly: int, lz: int) -> tuple[np.ndarray, bool, int]:
    topo = topologies.copy()
    flag = 1
    direction = np.array([[1, 1],
                          [1, -1],
                          [-1, 1],
                          [-1, -1]])
    changed_voxels = 0

    while flag:
        flag = 0
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    if topo[k, j, i] == 1:
                        for m in range(4):
                            if k + direction[m, 0] in range(lz) and j + direction[m, 1] in range(ly):
                                if topo[k + direction[m, 0], j + direction[m, 1], i] == 1:
                                    if topo[k + direction[m, 0], j, i] == 0 and topo[k, j + direction[m, 1], i] == 0:
                                        if random.getrandbits(1) == 1:
                                            topo[k + direction[m, 0], j, i] = 1
                                            changed_voxels += 1
                                        else:
                                            topo[k, j + direction[m, 1], i] = 1
                                            changed_voxels += 1
                                        flag = 1
                                        break

                            if k + direction[m, 0] in range(lz) and i + direction[m, 1] in range(lx):
                                if topo[k + direction[m, 0], j, i + direction[m, 1]] == 1:
                                    if topo[k + direction[m, 0], j, i] == 0 and topo[k, j, i + direction[m, 1]] == 0:
                                        if random.getrandbits(1) == 1:
                                            topo[k + direction[m, 0], j, i] = 1
                                            changed_voxels += 1
                                        else:
                                            topo[k, j, i + direction[m, 1]] = 1
                                            changed_voxels += 1
                                        flag = 1
                                        break

                            if j + direction[m, 0] in range(ly) and i + direction[m, 1] in range(lx):
                                if topo[k, j + direction[m, 0], i + direction[m, 1]] == 1:
                                    if topo[k, j + direction[m, 0], i] == 0 and topo[k, j, i + direction[m, 1]] == 0:
                                        if random.getrandbits(1) == 1:
                                            topo[k, j + direction[m, 0], i] = 1
                                            changed_voxels += 1
                                        else:
                                            topo[k, j, i + direction[m, 1]] = 1
                                            changed_voxels += 1
                                        flag = 1
                                        break

                            for z in [1, -1]:
                                if k + direction[m, 0] in range(lz) and j + direction[m, 1] in range(
                                        ly) and i + z in range(lx):
                                    if topo[k + direction[m, 0], j + direction[m, 1], i + z] == 1:
                                        if topo[k + direction[m, 0], j, i] == 0 and topo[
                                            k, j + direction[m, 1], i] == 0 and topo[k, j, i + z] == 0:
                                            if topo[k + direction[m, 0], j + direction[m, 1], i] == 0 and topo[
                                                k + direction[m, 0], j, i + z] == 0 and topo[
                                                k, j + direction[m, 1], i + z] == 0:
                                                rand1 = random.randint(1, 3)
                                                rand2 = random.randint(1, 2)

                                                if rand1 == 1:
                                                    topo[k + direction[m, 0], j, i] = 1
                                                    changed_voxels += 1
                                                    if rand2 == 1:
                                                        topo[k + direction[m, 0], j + direction[m, 1], i] = 1
                                                        changed_voxels += 1
                                                    else:
                                                        topo[k + direction[m, 0], j, i + z] = 1
                                                        changed_voxels += 1

                                                if rand1 == 2:
                                                    topo[k, j + direction[m, 1], i] = 1
                                                    changed_voxels += 1
                                                    if rand2 == 1:
                                                        topo[k + direction[m, 0], j + direction[m, 1], i] = 1
                                                        changed_voxels += 1

                                                    else:
                                                        topo[k, j + direction[m, 1], i + z] = 1
                                                        changed_voxels += 1

                                                if rand1 == 3:
                                                    topo[k, j, i + z] = 1
                                                    changed_voxels += 1
                                                    if rand2 == 1:
                                                        topo[k + direction[m, 0], j, i + z] = 1
                                                        changed_voxels += 1
                                                    else:
                                                        topo[k, j + direction[m, 1], i + z] = 1
                                                        changed_voxels += 1
                                                flag = 1
    return topo, np.array_equal(topologies, topo), changed_voxels


# @njit  # Currently not in use. Instead, make_voxels_surface_contact is used.
# def connect_island(local_arr: np.ndarray, labeled_arr: np.ndarray, max_label: int,
#                    x_start_gap: int, y_start_gap: int, z_start_gap: int) -> tuple[np.ndarray, int]:
#     arr_copy = local_arr.copy()
#     llx = local_arr.shape[0]
#     lly = local_arr.shape[1]
#     llz = local_arr.shape[2]
#     changed_voxels = 0
#     main_label = labeled_arr[1 - x_start_gap, 1 - y_start_gap, 1 - z_start_gap]
#     other_labels = set([label_idx for label_idx in range(1, max_label + 1)])
#     other_labels.remove(main_label)
#     for i in range(llx):
#         for j in range(lly):
#             for k in range(llz):
#                 if labeled_arr[i, j, k] in other_labels:
#                     changed_voxels += 1
#                     arr_copy[i, j, k] = 0
#         return arr_copy, changed_voxels


# @njit
# def add_random_voxels(arr_3d, probability=0.01):
#     arr_copy = arr_3d.copy()
#     changed_voxels = 0
#     lx = arr_3d.shape[0]
#     ly = arr_3d.shape[1]
#     lz = arr_3d.shape[2]
#
#     for i in range(lx):
#         for j in range(ly):
#             for k in range(lz):
#                 if (not arr_copy[i, j, k]) and (np.random.random() < probability):
#                     arr_copy[i, j, k] = 1
#                     changed_voxels += 1
#     return arr_copy, changed_voxels


# def generate_gaussian_rv(shape, variance):
#     return multivariate_normal(mean=[(shape[0] - 1) / 2, (shape[1] - 1) / 2, (shape[2] - 1) / 2],
#                                cov=[[variance, 0, 0], [0, variance, 0], [0, 0, variance]])
#
#
# def generate_gaussian_rv_pdf_arr(shape, variance):
#     gaussian_rv = generate_gaussian_rv(shape, variance)
#     arr = np.empty(shape)
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             for k in range(shape[2]):
#                 arr[i, j, k] = gaussian_rv.pdf((i, j, k))
#     return arr
#
#
# def add_gaussian_random_voxels(arr_3d, arr_3d_pdf, probability_factor):
#     arr_copy = arr_3d.copy()
#     lx = arr_3d.shape[0]
#     ly = arr_3d.shape[1]
#     lz = arr_3d.shape[2]
#     for i in range(lx):
#         for j in range(ly):
#             for k in range(lz):
#                 arr_copy[i, j, k] = np.random.binomial(lx * ly * lz * probability_factor, arr_3d_pdf[i, j, k])
#     return arr_copy


# def random_gauss_3d_array(shape):
#     lx = shape[0]
#     ly = shape[1]
#     lz = shape[2]
#     variance = 1
#     rv = multivariate_normal(mean=[(lx - 1)/2, (ly - 1)/2, (lz - 1)/2],
#                              cov=[[variance, 0, 0], [0, variance, 0], [0, 0, variance]])
#     arr = np.zeros(shape)
#     for i in range(lx):
#         for j in range(ly):
#             for k in range(lz):
#                 p = rv.pdf((i, j, k))
#                 arr[i, j, k] = np.random.binomial(lx * ly * lz, p)
#     return arr
