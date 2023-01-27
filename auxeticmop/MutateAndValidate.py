from scipy.ndimage import label
from numba import njit, int32
import numpy as np


def make_3d_print_without_support(arr_3d: np.ndarray, max_distance: int = 1) -> int:
    x_size, y_size, z_size = arr_3d.shape
    total_changed_voxels = 0
    while True:
        changed_voxels = 0
        for y_direction in (1, -1):
            for y_idx in range(1, y_size) if y_direction == 1 else range(y_size - 2, -1, -1):
                # Declaration of arr_2d_quarter and labeled_arr
                labeled_arr, max_island_idx = label(arr_3d[:, y_idx, :])
                dead_islands, survived_islands = dead_and_survived_islands(y_idx=y_idx, y_direction=y_direction,
                                                                           x_size=x_size, z_size=z_size,
                                                                           max_island_idx=max_island_idx,
                                                                           labeled_arr=labeled_arr, arr_3d=arr_3d)
                # Eliminating bad voxels by dead islands and survived islands with bad angle
                changed_voxels = voxel_elimination_by_islands(
                    x_size=x_size, z_size=z_size, labeled_arr=labeled_arr, dead_islands=np.array(list(dead_islands)),
                    survived_islands=np.array(list(survived_islands)), arr_3d=arr_3d, y_idx=y_idx,
                    max_distance=max_distance, y_direction=y_direction)
                total_changed_voxels += changed_voxels
        if changed_voxels == 0:
            break
    return total_changed_voxels


@njit
def dead_and_survived_islands(y_idx: int, y_direction: int, x_size: int, z_size: int, max_island_idx: int,
                              labeled_arr: np.ndarray, arr_3d: np.ndarray) -> tuple[set, set]:
    survived_islands = set()
    dead_islands = set()
    # Determining which ones are the dead islands
    for x_idx in range(x_size):
        for z_idx in range(z_size):
            island_label_num = labeled_arr[x_idx, z_idx]
            if island_label_num and arr_3d[x_idx, y_idx - y_direction, z_idx]:
                survived_islands.add(island_label_num)
    for island_label_num in range(1, max_island_idx + 1):
        if island_label_num not in survived_islands:
            dead_islands.add(island_label_num)
    return dead_islands, survived_islands


@njit
def voxel_elimination_by_islands(x_size: int, z_size: int, labeled_arr: np.ndarray, dead_islands: np.ndarray,
                                 survived_islands: np.ndarray, arr_3d: np.ndarray, y_idx: int,
                                 max_distance: int, y_direction: int) -> int:
    changed_voxels = 0
    for x_idx in range(x_size):
        for z_idx in range(z_size):
            island_idx = labeled_arr[x_idx, z_idx]
            if island_idx in dead_islands:
                arr_3d[x_idx, y_idx, z_idx] = 0
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
                        if arr_3d[x_position, y_idx - y_direction, z_position]:
                            is_any_around = True
                if not is_any_around:
                    # print(f'Overhang island Eliminated (x,y,z) = ({x_idx},{y_idx},{z_idx})')
                    arr_3d[x_idx, y_idx, z_idx] = 0
                    changed_voxels -= 1
    return changed_voxels


def one_connected_tree(arr_3d: np.ndarray) -> int | None:
    arr_shape = arr_3d.shape
    labeled_arr, max_label_idx = label(arr_3d)
    survived_labels = survived_tree_labels(labeled_arr, arr_shape[0], arr_shape[1], arr_shape[2])
    if len(survived_labels) == 0:
        return None
    else:
        unique_arr = np.array(np.unique(labeled_arr, return_counts=True))
        unique_sub_arr = unique_arr[:, np.isin(unique_arr[0], list(map(int, survived_labels)))]
        survived_label = unique_sub_arr[0, np.argmax(unique_sub_arr[1])]
        changed_voxels = one_survived_tree(arr_3d=arr_3d, labeled_arr=labeled_arr, survived_label=survived_label)
        return changed_voxels


@njit
def survived_tree_labels(labeled_arr: np.ndarray, lx: int, ly: int, lz: int) -> set:
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
    return labels_plane_1.intersection(labels_plane_2).intersection(labels_plane_3).intersection(
        labels_plane_4).intersection(labels_plane_5).intersection(labels_plane_6).difference({int32(0)})


@njit
def one_survived_tree(arr_3d: np.ndarray, labeled_arr: np.ndarray, survived_label: int) -> int:
    lx = arr_3d.shape[0]
    ly = arr_3d.shape[1]
    lz = arr_3d.shape[2]
    changed_voxels = 0
    for idx_1 in range(lx):
        for idx_2 in range(ly):
            for idx_3 in range(lz):
                if labeled_arr[idx_1, idx_2, idx_3] == 0:
                    continue
                elif labeled_arr[idx_1, idx_2, idx_3] != survived_label:
                    changed_voxels -= 1
                    arr_3d[idx_1, idx_2, idx_3] = 0
    return changed_voxels


def mutate_and_validate_topology(arr_3d: np.ndarray, mutation_probability: float) -> None | np.ndarray:
    arr_3d_mutated, voxels_mutation = mutation(arr_3d.copy(), mutation_probability=mutation_probability)
    lx, ly, lz = arr_3d.shape
    while True:
        voxels_validation = 0
        while True:
            _arr_3d_mutated = arr_3d_mutated.copy()
            voxels_oct = one_connected_tree(arr_3d_mutated)
            if voxels_oct is None:
                return None
            voxels_vsc = make_voxels_surface_contact(arr_3d_mutated)
            voxels_3pws = make_3d_print_without_support(arr_3d_mutated)
            voxels_validation += voxels_oct + voxels_3pws + voxels_vsc
            if np.array_equal(arr_3d_mutated, _arr_3d_mutated):
                if voxels_oct == 0 and voxels_3pws == 0 and voxels_vsc == 0:
                    total_voxels_change = voxels_mutation + voxels_validation
                    print('> Total changes in structure during validation < ')
                    print("- Voxel number changes: [{}] voxels".format(total_voxels_change), end=' | ')
                    print("Changed of volume fraction: [{:.2f}] percents".format(
                        100 * total_voxels_change / (lx * ly * lz)))
                    return arr_3d_mutated
                else:
                    arr_3d_mutated, voxels_mutation = mutation(arr_3d, mutation_probability=mutation_probability)
                    break


@njit
def mutation(arr_3d: np.ndarray, mutation_probability: float) -> tuple[np.ndarray, int]:
    _arr_3d = arr_3d.copy()
    lx, ly, lz = arr_3d.shape
    changed_voxels = 0
    for idx_1 in range(lx):
        for idx_2 in range(ly):
            for idx_3 in range(lz):
                if np.random.random() < mutation_probability:
                    if _arr_3d[idx_1, idx_2, idx_3] == 1:
                        _arr_3d[idx_1, idx_2, idx_3] = 0
                        changed_voxels -= 1
                    else:
                        _arr_3d[idx_1, idx_2, idx_3] = 1
                        changed_voxels += 1
    return _arr_3d, changed_voxels


@njit
def make_voxels_surface_contact(arr_3d: np.ndarray) -> int:
    nx, ny, nz = arr_3d.shape
    total_changed_voxels = 0
    point_to_point = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
                               [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]])
    edge_to_edge = np.array([[0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1], [1, 0, 1], [-1, 0, 1],
                             [1, 0, -1], [-1, 0, -1], [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]])
    while True:
        changed_voxels = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if arr_3d[x, y, z] == 0:
                        continue
                    for xx, yy, zz in point_to_point:
                        px, py, pz = x + xx, y + yy, z + zz
                        if px < 0 or py < 0 or pz < 0 or px == nx or py == ny or pz == nz:
                            continue
                        if arr_3d[px, py, pz] == 0:
                            continue
                        if arr_3d[px, y, z] or arr_3d[x, py, z] or arr_3d[x, y, pz]:  # 100 010 001
                            continue
                        if arr_3d[px, py, z] or arr_3d[px, y, pz] or arr_3d[x, py, pz]:  # 110 101 011
                            continue
                        routine = np.random.randint(6)
                        changed_voxels += 2
                        if routine == 0:
                            arr_3d[px, y, z] = 1  # 100
                            arr_3d[px, py, z] = 1  # 110
                        elif routine == 1:
                            arr_3d[px, y, z] = 1  # 100
                            arr_3d[px, y, pz] = 1  # 101
                        elif routine == 2:
                            arr_3d[x, py, z] = 1  # 010
                            arr_3d[px, py, z] = 1  # 110
                        elif routine == 3:
                            arr_3d[x, py, z] = 1  # 010
                            arr_3d[x, py, pz] = 1  # 011
                        elif routine == 4:
                            arr_3d[x, y, pz] = 1  # 001
                            arr_3d[px, y, pz] = 1  # 101
                        else:
                            arr_3d[x, y, pz] = 1  # 001
                            arr_3d[x, py, pz] = 1  # 011
                    for xx, yy, zz in edge_to_edge:
                        px, py, pz = x + xx, y + yy, z + zz
                        if px < 0 or py < 0 or pz < 0 or px == nx or py == ny or pz == nz:
                            continue
                        if arr_3d[px, py, pz] == 0:
                            continue
                        if xx == 0:  # 011
                            if arr_3d[x, y, pz] or arr_3d[x, py, z]:
                                continue
                            routine = np.random.randint(2)
                            changed_voxels += 1
                            if routine == 0:
                                arr_3d[x, y, pz] = 1  # 001
                            else:
                                arr_3d[x, py, z] = 1  # 010
                        elif yy == 0:  # 101
                            if arr_3d[px, y, z] or arr_3d[x, y, pz]:
                                continue
                            routine = np.random.randint(2)
                            changed_voxels += 1
                            if routine == 0:
                                arr_3d[px, y, z] = 1  # 100
                            else:
                                arr_3d[x, y, pz] = 1  # 001
                        else:  # 110
                            if arr_3d[px, y, z] or arr_3d[x, py, z]:
                                continue
                            routine = np.random.randint(2)
                            changed_voxels += 1
                            if routine == 0:
                                arr_3d[px, y, z] = 1  # 100
                            else:
                                arr_3d[x, py, z] = 1  # 010
        total_changed_voxels += changed_voxels
        if changed_voxels == 0:
            break
    return total_changed_voxels
