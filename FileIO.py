import numpy as np
import pandas as pd
import os


def read_numpy_from_csv(file_name, w, from_type, to_type):
    return pd.read_csv(f'./{file_name}_{w}.csv', header=None, dtype=from_type).to_numpy(dtype=to_type)


def array_to_csv(path, arr, dtype, mode, save_as_int=False):
    if mode == 'a' and os.path.isfile(path):
        previous_arr = np.genfromtxt(path, delimiter=',', dtype=dtype)
        # print('[array_to_csv] append shape: ', previous_arr.shape, arr.shape)
        arr = np.vstack((previous_arr, arr))
    fmt = '%i' if save_as_int else '%.18e'
    np.savetxt(path, arr, delimiter=',', fmt=fmt)


def offspring_import(w, mode):
    if mode == 'Random':
        topo_1 = np.genfromtxt('topo_parent_' + str(w) + '.csv', delimiter=',', dtype=int)
    else:
        topo_1 = np.genfromtxt('topo_offspring_' + str(w) + '.csv', delimiter=',', dtype=int)
    result_1 = np.genfromtxt('Output_offspring_' + str(w) + '.csv', delimiter=',', dtype=np.float32)
    return topo_1, result_1


def parent_import(w, restart_pop):
    topo = np.genfromtxt('topo_parent_' + str(w) + '.csv', delimiter=',', dtype=int)
    reslt = np.genfromtxt('Output_parent_' + str(w) + '.csv', delimiter=',', dtype=np.float32)

    if restart_pop == 0:
        return topo, reslt

    else:  # restart
        offspring = np.genfromtxt('topo_offspring_' + str(w) + '.csv', delimiter=',', dtype=int)
        return topo, reslt, offspring


def parent_export(w, next_generations, end_pop, results, results_1, topologies, topologies_1):
    for i in next_generations:
        if i < end_pop:
            array_to_csv(f'Output_parent_{w + 1}.csv', results[i], dtype=np.float32, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topologies[i], dtype=int, mode='a', save_as_int=True)

        if i >= end_pop:
            array_to_csv(f'Output_parent_{w + 1}.csv', results_1[i - end_pop], dtype=np.float32, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topologies_1[i - end_pop], dtype=int, mode='a', save_as_int=True)
