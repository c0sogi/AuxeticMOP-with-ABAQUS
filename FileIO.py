import numpy as np
import pandas as pd
import os


def read_numpy_from_csv(file_name, w, from_type, to_type):
    return pd.read_csv(f'./{file_name}_{w}.csv', header=None, dtype=from_type).to_numpy(dtype=to_type)


def array_to_csv(path, arr, dtype, mode, save_as_int=False):
    if mode == 'a' and os.path.isfile(path):
        previous_arr = np.genfromtxt(path, delimiter=',', dtype=dtype)
        arr = np.vstack((previous_arr, arr))
    fmt = '%i' if save_as_int else '%.18e'
    np.savetxt(path, arr, delimiter=',', fmt=fmt)


def offspring_import(w):
    topo_offspring = np.genfromtxt('topo_offspring_' + str(w) + '.csv', delimiter=',', dtype=int)
    result_offspring = np.genfromtxt('Output_offspring_' + str(w) + '.csv', delimiter=',', dtype=float)
    return topo_offspring, result_offspring


def parent_import(w):
    topo_parent = np.genfromtxt('topo_parent_' + str(w) + '.csv', delimiter=',', dtype=int)
    result_parent = np.genfromtxt('Output_parent_' + str(w) + '.csv', delimiter=',', dtype=float)
    return topo_parent, result_parent


def parent_export(w, next_generations, population_size,
                  result_parent, result_offspring, topo_parent, topo_offspring):
    for i in next_generations:
        if i < population_size:
            array_to_csv(f'Output_parent_{w + 1}.csv', result_parent[i], dtype=float, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topo_parent[i],
                         dtype=int, mode='a', save_as_int=True)

        if i >= population_size:
            array_to_csv(f'Output_parent_{w + 1}.csv', result_offspring[i - population_size], dtype=float, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topo_offspring[i - population_size],
                         dtype=int, mode='a', save_as_int=True)
