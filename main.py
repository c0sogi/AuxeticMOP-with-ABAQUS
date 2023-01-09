import os
import pickle
import numpy as np
from time import sleep
from datetime import datetime
from GraphicUserInterface import App
from multiprocessing import Process, Pipe
from dataclasses import asdict
from GeneticAlgorithm import generate_offspring
from PostProcessing import evaluation, visualize2, selection
from FileIO import parent_import, parent_export, offspring_import


def make_and_start_process(target, duplex=True, daemon=True):
    conn_1, conn_2 = Pipe(duplex=duplex)
    process = Process(target=target, args=(conn_2,), daemon=daemon)
    process.start()
    return process, conn_1, conn_2


def remove_file(file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)


def open_abaqus(abaqus_script_name, directory, params, abaqus_execution_mode):
    with open('./PARAMS', mode='wb') as f_params:
        pickle.dump({**asdict(params), **{'setPath': directory}}, f_params, protocol=2)
    print(f"========== Opening ABAQUS on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}! ==========")
    process = Process(target=os.system, args=(f'abaqus cae {abaqus_execution_mode}={abaqus_script_name}',), daemon=True)
    process.start()
    return process


def wait_for_abaqus_to_complete(check_exit_time, restart, w, offspring):
    args = {
        'restart': restart,
        'w': w,
        'offspring': np.swapaxes(offspring, axis1=1, axis2=3)
        # Changing offspring axis: (end_pop, lx, ly, lz) -> (end_pop, lz, ly, lx)
    }
    with open('./args', mode='wb') as f_args:
        pickle.dump(args, f_args, protocol=2)
    print('Waiting for abaqus')
    while True:
        sleep(check_exit_time)
        if os.path.isfile('./args'):
            print('.', end='')
            continue
        else:
            print()
            break
    print(f"========== An abaqus job done on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}! ==========")


def one_generation(w, restart, p_conn, params):
    if restart:
        topologies, results, offspring = parent_import(w=w, restart_pop=params.restart_pop)
        offspring = offspring.reshape((params.end_pop, params.lx, params.ly, params.lz))
    else:
        topologies, results = parent_import(w=w, restart_pop=0)
        offspring = generate_offspring(topologies=topologies, w=w, end_pop=params.end_pop,
                                       timeout=params.timeout, mutation_rate=params.mutation_rate,
                                       lx=params.lx, ly=params.ly, lz=params.lz)
    wait_for_abaqus_to_complete(check_exit_time=1, restart=restart, w=w, offspring=offspring)

    topologies_1, results_1 = offspring_import(w=w, mode=params.mode)
    fitness_values = evaluation(topo=topologies, topo_1=topologies_1, reslt=results, reslt_1=results_1,
                                lx=params.lx, ly=params.ly, lz=params.lz, max_rf22=params.MaxRF22,
                                evaluation_version=params.evaluation_version, q=params.end_pop,
                                penalty_coefficient=params.penalty_coefficient)

    pop, next_generations = selection(pop=np.append(topologies, topologies_1, axis=0), fitness_values=fitness_values,
                                      pop_size=params.end_pop)
    parent_export(topologies=topologies, topologies_1=topologies_1, results=results, results_1=results_1,
                  w=w, end_pop=params.end_pop, next_generations=next_generations)
    if restart:
        params.restart_pop = 0
    print('iteration:', w)
    visualize2(w=w, lx=params.lx, ly=params.ly, lz=params.lz,
               penalty_coefficient=params.penalty_coefficient, evaluation_version=params.evaluation_version,
               max_rf22=params.MaxRF22, parent_conn=p_conn, file_io=True)


def plot_previous_data(conn_1):
    if os.path.isfile(f'Plot_data'):
        with open(f'Plot_data', mode='rb') as f_read:
            read_data = pickle.load(f_read)
        for key, value in read_data.items():
            conn_1.send(value)


if __name__ == '__main__':
    # Make an interface and receive parameters
    gui_process, parent_conn, child_conn = make_and_start_process(target=App, duplex=True, daemon=True)
    set_path, parameters = parent_conn.recv()
    parameters.post_initialize()
    os.chdir(set_path)

    # load previous plotting data
    remove_file(file_name='args')
    plot_previous_data(conn_1=parent_conn)

    # Open an abaqus process
    abaqus_process = open_abaqus(abaqus_script_name=parameters.abaqus_script_name, params=parameters,
                                 abaqus_execution_mode=parameters.abaqus_execution_mode, directory=set_path)

    # Start working
    try:
        if parameters.mode == 'GA':
            if parameters.restart_pop == 0:
                for gen_idx in range(parameters.ini_gen, parameters.end_gen + 1):
                    one_generation(w=gen_idx, restart=False, params=parameters, p_conn=parent_conn)
            else:
                one_generation(w=parameters.ini_gen, restart=True, params=parameters,
                               p_conn=parent_conn)
                for gen_idx in range(parameters.ini_gen + 1, parameters.end_gen + 1):
                    one_generation(w=gen_idx, restart=False, params=parameters, p_conn=parent_conn)
        elif parameters.mode == 'Something':
            pass

        # Make abaqus exit itself
        with open('args_end', mode='wb') as f_args_end:
            pickle.dump('end', f_args_end, protocol=2)
    except Exception as error_message:
        # An error message
        print('[MAIN] An error in main function occurred while generating generations: \n', error_message)
    finally:
        # Clear processes
        gui_process.kill()
        abaqus_process.kill()
        parent_conn.close()
        child_conn.close()
