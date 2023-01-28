import os
import pickle
import asyncio
import aiofiles
import numpy as np
import re
from typing import Union


def remove_file(file_name: str) -> None:
    """
    Remove a file with designated filename.
    :param file_name: The filename to remove.
    :return: None
    """
    if os.path.isfile(file_name):
        os.remove(file_name)


def key_modifier(key: str, option: Union[str, None]) -> object:
    if option == 'int':
        return int(re.compile(r'\d+').search(key).group())
    else:
        return key


def get_sorted_file_numbers_from_pattern(p):
    return sorted(
        [int(re.compile(r'\d+').search(s).group()) for s in [f for f in os.listdir() if re.compile(p).match(f)]])


def find_job_location_from_offspring(params_dict):
    from .PostProcessing import evaluate_all_fitness_values, find_pareto_front_points
    from .ParameterDefinitions import fitness_definitions

    def _find_job_location_from_offspring(g, prt_idx, tp_offs, tp_prt, tp_pf):
        arg = np.argwhere(np.all(tp_offs == tp_prt, axis=(2,)))
        if len(arg) == 0:
            arg = np.argwhere(np.all(tp_pf == tp_prt, axis=(1,)))
            print(f'Pareto topo in parent {g} - {prt_idx + 1} is in parent ', 1, '-', arg[0, 0] + 1)
        else:
            print(f'Pareto topo in parent {g} - {prt_idx + 1} is in offspring ', arg[0, 0] + 1, '-',
                  arg[0, 1] + 1)

    file_headers = ('Topologies', 'FieldOutput_offspring', 'FieldOutput')
    topo_parent, topo_offspring, result_parent, result_offspring = None, None, dict(), dict()
    for file_header in file_headers:
        file_numbers = get_sorted_file_numbers_from_pattern(rf'{file_header}_\d+')
        file_names = [f'{file_header}_{file_number}' for file_number in file_numbers]
        loaded_dict = asyncio.run(pickles_aio(file_names, mode='r', key_option='int'))
        for gen_num in loaded_dict.keys():
            if 'parent' in loaded_dict[gen_num]:
                if topo_parent is None:
                    topo_parent = np.expand_dims(loaded_dict[gen_num]['parent'], axis=0)
                else:
                    topo_parent = np.vstack((topo_parent, np.expand_dims(loaded_dict[gen_num]['parent'], axis=0)))
            if 'offspring' in loaded_dict[gen_num]:
                if topo_offspring is None:
                    topo_offspring = np.expand_dims(loaded_dict[gen_num]['offspring'], axis=0)
                else:
                    topo_offspring = np.vstack(
                        (topo_offspring, np.expand_dims(loaded_dict[gen_num]['offspring'], axis=0)))
            else:
                if file_header == 'FieldOutput':
                    result_parent.update({gen_num: loaded_dict[gen_num]})
                elif file_header == 'FieldOutput_offspring':
                    result_offspring.update({gen_num: loaded_dict[gen_num]})
    for gen in get_sorted_file_numbers_from_pattern(r'FieldOutput_\d+'):
        fit_vals = evaluate_all_fitness_values(params_dict=params_dict, results=result_parent[gen],
                                               topologies=topo_parent[gen - 1], fitness_definitions=fitness_definitions)
        pareto_indices = find_pareto_front_points(costs=fit_vals, return_index=True)
        print(f'\n>>> Parent {gen} <<<')
        print('Pareto topos: ', pareto_indices + 1)
        for pareto_index in pareto_indices:
            topo_pareto = topo_parent[gen - 1][pareto_index]
            _find_job_location_from_offspring(g=gen, prt_idx=pareto_index, tp_offs=topo_offspring,
                                              tp_prt=topo_pareto, tp_pf=topo_parent[0])


# def dump_pickled_dict_data(file_name: str, key: object, to_dump: object, mode: str) -> None:
#     if mode == 'a' and os.path.isfile(file_name):
#         with open(file_name, mode='rb') as f:
#             dict_data = pickle.load(f, encoding='latin1')
#         dict_data.update({key: to_dump})
#     else:
#         dict_data = {key: to_dump}
#     with open(file_name, mode='wb') as f:
#         pickle.dump(dict_data, f, protocol=2)
#
#
# def load_pickled_dict_data(file_name: str) -> dict:
#     with open(file_name, mode='rb') as f:
#         dict_data = pickle.load(f, encoding='latin1')
#     return dict_data


async def pickle_aio(file_name: str, mode: str, to_dump: Union[dict, list, np.ndarray] = None) -> any:
    encoding = 'latin1'
    if mode == 'r':
        async with aiofiles.open(file_name, mode='rb') as f:
            serialized_pickle = await f.read()
        print(file_name, 'loaded!')
        return pickle.loads(serialized_pickle, encoding=encoding)
    elif mode == 'a' and os.path.isfile(file_name):
        async with aiofiles.open(file_name, mode='rb') as f:
            serialized_pickle = await f.read()
        loaded = pickle.loads(serialized_pickle, encoding=encoding)
        if isinstance(loaded, dict):
            loaded.update(to_dump)
        elif isinstance(loaded, list):
            loaded += to_dump
        elif isinstance(loaded, np.ndarray):
            loaded = np.vstack((loaded, to_dump))
        else:
            raise ValueError
        serialized_pickle = pickle.dumps(loaded, protocol=2)
        async with aiofiles.open(file_name, mode='wb') as f:
            await f.write(serialized_pickle)
        print(file_name, 'dumped!')
    elif mode == 'w' or mode == 'a':
        serialized_pickle = pickle.dumps(to_dump, protocol=2)
        async with aiofiles.open(file_name, mode='wb') as f:
            await f.write(serialized_pickle)
        print(file_name, 'dumped!')
    else:
        raise ValueError


async def pickles_aio(file_names: Union[list, tuple], mode: str, to_dumps=None, key_option=None) -> dict:
    if mode == 'r':
        loaded = await asyncio.gather(*[asyncio.ensure_future(pickle_aio(file_name, mode=mode))
                                        for file_name in file_names])
        return {key_modifier(key, option=key_option): value for key, value in zip(file_names, loaded)}
    else:
        await asyncio.gather(*[asyncio.ensure_future(pickle_aio(file_name, mode=mode, to_dump=to_dump))
                               for to_dump, file_name in zip(to_dumps, file_names)])


def pickle_io(file_name: str, mode: str, to_dump: object = None) -> any:
    return asyncio.run(pickle_aio(file_name=file_name, mode=mode, to_dump=to_dump))


def pickles_io(file_names: Union[list, tuple], mode: str, to_dumps=None, key_option=None) -> dict:
    return asyncio.run(pickles_aio(file_names=file_names, mode=mode, to_dumps=to_dumps, key_option=key_option))


# def open_history_output(gen, path=None):
#     if path is not None:
#         from os import chdir
#         chdir(path)
#     from pickle import load
#     file_names = ('U1_HistoryOutput', 'U2_HistoryOutput', 'U3_HistoryOutput',
#                   'RF1_HistoryOutput', 'RF2_HistoryOutput', 'RF3_HistoryOutput')
#     for file_name in file_names:
#         pickle_file_name = file_name + f'_{gen}'
#         with open(pickle_file_name, 'rb') as f:
#             loaded_file = load(f, encoding='bytes')
#             print(f'Gen{gen}-{pickle_file_name}: {loaded_file}')
