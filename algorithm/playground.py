import numpy as np
import random
from functools import reduce
from itertools import product
from scipy.ndimage import label, distance_transform_edt, distance_transform_cdt, distance_transform_bf
from topologyValidation import visualize_one_cube


def random_array(shape):
    return np.random.randint(2, size=reduce(lambda x, y: x * y, shape)).reshape(shape)


def is_vertex_contact(arr_3d):
    arr_copy = arr_3d.copy()
    lx = arr_3d.shape[0]
    ly = arr_3d.shape[1]
    lz = arr_3d.shape[2]

# arr = np.array([[[0, 1, 1], [1, 0, 0], [1, 1, 0]],
#                 [[0, 1, 0], [0, 0, 1], [1, 1, 0]],
#                 [[1, 1, 1], [0, 0, 0], [1, 1, 1]]])


def distance_between_vertexs(v):
    return np.linalg.norm(v[0] - v[1])


# while True:
#     arr333 = random_array((3, 3, 3))
#     arr333[1, 1, 1] = 1
#     arr333_copy = arr333.copy()
#     labeled_arr, max_label = label(arr333)
#     # print(labeled_arr, max_label)
#     if max_label <= 1:
#         print('No jam')
#         continue
#     else:
#         main_label = labeled_arr[1, 1, 1]
#         labeled_arr = np.array(labeled_arr)
#         print(set([label_idx for label_idx in range(1, max_label + 1)]))
#         other_labels = set([label_idx for label_idx in range(1, max_label + 1)])
#         other_labels.discard(main_label)
#         for other_label in other_labels:
#             adjacent_cell = np.where(labeled_arr == other_label)
#             np.max
#             print(f'Adject cell of label {other_label}: {adjacent_cell[0], adjacent_cell[1], adjacent_cell[2]}')
#             print('dstack', np.dstack([adjacent_cell[0], adjacent_cell[1], adjacent_cell[2]])[0])
#             print('max', max(np.dstack([adjacent_cell[0], adjacent_cell[1], adjacent_cell[2]])[0], key=distance_between_vertexs))
#             # print(max(, key=distance_between_vertexs))
#         break

# arr333 = random_array((3,3,3))
# arr333 = np.array([[[0, 0, 0],[0, 1, 1],[0, 1, 1]], [[0, 0, 0], [0, 1, 1], [0, 1, 1]], [[1, 0, 0], [0, 0, 0], [0, 0, 0]]])
#
#
# arr333 = np.ones((3,3,3))
# local_arr = arr333[1:, 1:, 1:]
# arr333[1:, 1:, 1:] = np.zeros((2,2,2))
# print(arr333)

# path = rf'C:\Users\dcas\PythonCodes\Coop\pythoncode\10x10x10 - 복사본\topo_parent_1.csv'
# cube_4d_array = np.genfromtxt(path, dtype=int, delimiter=',').reshape((100, 10, 10, 10))
# for arr_3d in cube_4d_array:
#     visualize_one_cube(arr_3d)

import pickle
path = r'C:\Users\dcas\PythonCodes\Coop\pythoncode\22-12-31\data\args'
with open(path, mode='rb') as f:
    a = pickle.load(f)

print(a['offspring'].shape)