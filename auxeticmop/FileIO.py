import numpy as np
import os
import pickle


def array_to_csv(path, arr, dtype, mode, save_as_int=False):
    if mode == 'a' and os.path.isfile(path):
        previous_arr = np.genfromtxt(path, delimiter=',', dtype=dtype)
        arr = np.vstack((previous_arr, arr))
    fmt = '%i' if save_as_int else '%.18e'
    np.savetxt(path, arr, delimiter=',', fmt=fmt)


def offspring_import(gen_num):
    topo_offspring = np.genfromtxt('topo_offspring_' + str(gen_num) + '.csv', delimiter=',', dtype=int)
    result_offspring = np.genfromtxt('Output_offspring_' + str(gen_num) + '.csv', delimiter=',', dtype=float)
    return topo_offspring, result_offspring


def parent_import(gen_num):
    try:
        topo_parent = np.genfromtxt('topo_parent_' + str(gen_num) + '.csv', delimiter=',', dtype=int)
    except FileNotFoundError:
        topo_parent = None
    try:
        result_parent = np.genfromtxt('Output_parent_' + str(gen_num) + '.csv', delimiter=',', dtype=float)
    except FileNotFoundError:
        result_parent = None
    return topo_parent, result_parent


def parent_export(gen_num, next_generations, population_size,
                  result_parent, result_offspring, topo_parent, topo_offspring):
    for i in next_generations:
        if i < population_size:
            dump_pickled_dict_data(f'Topologies_{gen_num + 1}', key='parent', to_dump=result_parent[i], mode='a')
            dump_pickled_dict_data(f'FieldOutput_{gen_num + 1}', key='parent', to_dump=result_parent[i], mode='a')
            array_to_csv(f'Output_parent_{gen_num + 1}.csv', result_parent[i], dtype=float, mode='a')
            array_to_csv(f'topo_parent_{gen_num + 1}.csv', topo_parent[i],
                         dtype=int, mode='a', save_as_int=True)

        if i >= population_size:
            array_to_csv(f'Output_parent_{gen_num + 1}.csv', result_offspring[i - population_size], dtype=float,
                         mode='a')
            array_to_csv(f'topo_parent_{gen_num + 1}.csv', topo_offspring[i - population_size],
                         dtype=int, mode='a', save_as_int=True)


def dump_pickled_dict_data(file_name, key, to_dump, mode):
    if mode == 'a' and os.path.isfile(file_name):
        with open(file_name, mode='rb') as f:
            dict_data = pickle.load(f, encoding='latin1')
        dict_data.update({key: to_dump})
    else:
        dict_data = {key: to_dump}
    with open(file_name, mode='wb') as f:
        pickle.dump(dict_data, f, protocol=2)


def load_pickled_dict_data(file_name):
    with open(file_name, mode='rb') as f:
        dict_data = pickle.load(f, encoding='latin1')
    return dict_data
