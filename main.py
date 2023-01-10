import os
import pickle
import numpy as np
from time import sleep
from datetime import datetime
from GraphicUserInterface import App
from multiprocessing import Process, Pipe
from dataclasses import asdict
from GeneticAlgorithm import generate_offspring
from PostProcessing import evaluate_two_topologies, selection, Mop, get_hv_from_datum_hv
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


def one_generation(w, restart, params, mop):
    if restart:
        topo_parent, result_parent = parent_import(w=w)
        topo_offspring = np.genfromtxt('topo_offspring_' + str(w) + '.csv', delimiter=',', dtype=int)
        topo_offspring = topo_offspring.reshape((params.end_pop, params.lx, params.ly, params.lz))
    else:
        topo_parent, result_parent = parent_import(w=w)
        topo_offspring = generate_offspring(topo_parent=topo_parent, w=w, end_pop=params.end_pop,
                                            timeout=params.timeout, mutation_rate=params.mutation_rate,
                                            lx=params.lx, ly=params.ly, lz=params.lz)
    wait_for_abaqus_to_complete(check_exit_time=1, restart=restart, w=w, offspring=topo_offspring)

    topo_offspring, result_offspring = offspring_import(w=w)
    two_fit_vals = evaluate_two_topologies(topo_parent=topo_parent, topo_offspring=topo_offspring,
                                           result_parent=result_parent, result_offspring=result_offspring,
                                           lx=params.lx, ly=params.ly, lz=params.lz, max_rf22=params.max_rf22,
                                           evaluation_version=params.evaluation_version, population_size=params.end_pop,
                                           penalty_coefficient=params.penalty_coefficient)

    _, next_generations = selection(all_topologies=np.vstack((topo_parent, topo_offspring)),
                                    all_fitness_values=two_fit_vals, population_size=params.end_pop)
    parent_export(topo_parent=topo_parent, topo_offspring=topo_offspring,
                  result_parent=result_parent, result_offspring=result_offspring,
                  w=w, population_size=params.end_pop, next_generations=next_generations)
    if restart:
        params.restart_pop = 0
    print('Generation:', w)
    mop.visualize(params=params, w=w, use_manual_rp=False, ref_x=0.0, ref_y=0.0)


def plot_previous_data(conn_to_gui, use_manual_rp, ref_x=0.0, ref_y=0.0):
    if os.path.isfile('_plotting_data_'):
        with open('_plotting_data_', mode='rb') as f_read:
            read_data = pickle.load(f_read)
        all_datum_hv = np.empty((0,), dtype=float)
        all_lower_bounds = np.empty((0, 2), dtype=float)
        all_gens = np.empty((0,), dtype=int)
        _ref_x, _ref_y = 0.0, 0.0
        for generation_num in read_data.keys():
            pareto_1_sorted, pareto_2_sorted, datum_hv, lower_bounds = read_data[generation_num]
            all_datum_hv = np.hstack((all_datum_hv, datum_hv))
            all_lower_bounds = np.vstack((all_lower_bounds, lower_bounds))
            all_gens = np.hstack((all_gens, generation_num))
            if pareto_1_sorted[-1] > _ref_x:
                _ref_x = pareto_1_sorted[-1]
            if pareto_2_sorted[0] > _ref_y:
                _ref_y = pareto_2_sorted[0]
            if use_manual_rp:
                _ref_x, _ref_y = ref_x, ref_y
            all_hv = [get_hv_from_datum_hv(all_datum_hv[idx], all_lower_bounds[idx],
                                           ref_x=_ref_x, ref_y=_ref_y) for idx in range(len(all_gens))]

            print('')
            conn_to_gui.send((pareto_1_sorted, pareto_2_sorted, all_gens, all_hv))


if __name__ == '__main__':
    # Make an interface and receive parameters
    gui_process, parent_conn, child_conn = make_and_start_process(target=App, duplex=True, daemon=True)
    set_path, parameters = parent_conn.recv()
    parameters.post_initialize()
    os.chdir(set_path)

    # load previous plotting data
    remove_file(file_name='args')
    mop_for_plot = Mop(conn_to_gui=parent_conn)
    plot_previous_data(conn_to_gui=parent_conn, use_manual_rp=False)

    # Open an abaqus process
    abaqus_process = open_abaqus(abaqus_script_name=parameters.abaqus_script_name, params=parameters,
                                 abaqus_execution_mode=parameters.abaqus_execution_mode, directory=set_path)

    # Start working
    try:
        if parameters.mode == 'GA':
            if parameters.restart_pop == 0:
                for gen_idx in range(parameters.ini_gen, parameters.end_gen + 1):
                    one_generation(w=gen_idx, restart=False, params=parameters, mop=mop_for_plot)
            else:
                one_generation(w=parameters.ini_gen, restart=True, params=parameters, mop=mop_for_plot)
                for gen_idx in range(parameters.ini_gen + 1, parameters.end_gen + 1):
                    one_generation(w=gen_idx, restart=False, params=parameters, mop=mop_for_plot)
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
