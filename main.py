import os
import pickle
import numpy as np
from time import sleep
from datetime import datetime
import multiprocessing as mp
from multiprocessing import connection
from dataclasses import asdict
from GeneticAlgorithm import generate_offspring, random_parent_generation
from GraphicUserInterface import App, Visualizer, Parameters
from PostProcessing import evaluate_fitness_values, selection
from FileIO import parent_import, parent_export, offspring_import

ABAQUS_ARGUMENTS_FILENAME = 'args'
ABAQUS_PROCESS_END_FILENAME = 'args_end'
ABAQUS_PARAMETER_FILENAME = 'PARAMS'
PLOTTING_DATA_FILENAME = '_plotting_data_'


def make_and_start_process(target: any, duplex: bool = True,
                           daemon: bool = True) -> tuple[mp.Process, connection.Connection, connection.Connection]:
    """
    Make GUI process and return a process and two Pipe connections between main process and GUI process.
    :param target: The GUI class to run as another process.
    :param duplex: If True, both receiving and sending data between main process and GUI process will be allowed.
    Otherwise, conn_1 is only allowed for receiving data and conn_2 is only allowed for sending data.
    :param daemon: If True, GUI process will be terminated when main process is terminated.
    Otherwise, GUI process will be orphan process.
    :return: A running process, Pipe connections of main process, GUI process, respectively.
    """
    conn_1, conn_2 = mp.Pipe(duplex=duplex)
    process = mp.Process(target=target, args=(conn_2,), daemon=daemon)
    process.start()
    return process, conn_1, conn_2


def remove_file(file_name: str) -> None:
    """
    Remove a file before beginning GA.
    :param file_name: The filename to remove.
    :return: None
    """
    if os.path.isfile(file_name):
        os.remove(file_name)


def open_abaqus(abaqus_script_name: str, directory: str, params: Parameters, abaqus_execution_mode: str) -> mp.Process:
    """
    Open an abaqus CAE process
    :param abaqus_script_name: Name of python script file for abaqus. For example, ABQ.py
    :param directory: The directory for abaqus to work, where abaqus script file and other csv files are located.
    :param params: Parameters retrieved from GUI.
    :param abaqus_execution_mode: 'noGUI' for abaqus non-gui mode, 'script' for abaqus gui mode.
    :return: Abaqus process.
    """
    with open(ABAQUS_PARAMETER_FILENAME, mode='wb') as f_params:  # Saving a parameter file for abaqus
        pickle.dump({**asdict(params), **{'setPath': directory}}, f_params, protocol=2)
    print(f"========== Opening ABAQUS on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}! ==========")
    process = mp.Process(target=os.system, args=(f'abaqus cae {abaqus_execution_mode}={abaqus_script_name}',),
                         daemon=True)
    process.start()  # Start abaqus
    return process


def wait_for_abaqus_until_complete(check_exit_time: float, restart: bool, w: int, offspring: np.ndarray) -> None:
    """
    Hold main process until one generation of abaqus job is done.
    :param check_exit_time: The time in seconds checking whether abaqus job is done.
    :param restart: Restarting evolution from unfinished generation previously done.
    :param w: current generation number.
    :param offspring: Topologies of an offspring of current generation.
    :return: Nothing
    """

    args = {  # arguments for abaqus job, args will be transferred to abaqus
        'restart': restart,  # True for continue working from population of unfinished generation previously done
        'w': w,  # Generation number
        'offspring': np.swapaxes(offspring, axis1=1, axis2=3)
        # Changing offspring axis: (end_pop, lx, ly, lz) -> (end_pop, lz, ly, lx)
    }
    with open(ABAQUS_ARGUMENTS_FILENAME, mode='wb') as f_args:  # Saving args file
        pickle.dump(args, f_args, protocol=2)
    print('Waiting for abaqus')
    while True:  # Checking if one generation of abaqus job is done
        sleep(check_exit_time)
        if os.path.isfile(ABAQUS_ARGUMENTS_FILENAME):  # If args file exists, abaqus job is not done yet
            print('.', end='')
            continue
        else:  # If args file doesn't exist, it means abaqus job is done
            print()
            break
    print(f"========== An abaqus job done on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}! ==========")


def one_generation(gen: int, restart: bool, params: Parameters, visualizer: Visualizer = None) -> None:
    """
    Evolution of one generation will be done.
    :param gen: Current generation number. (1~)
    :param restart: Restarting evolution from unfinished generation previously done.
    :param params: Parameters from GUI.
    :param visualizer: A class for visualizing every result of GA.
    :return: Nothing
    """
    # Import parent topologies and outputs of current generation
    topo_parent, result_parent = parent_import(gen_num=gen)
    if result_parent is None:
        wait_for_abaqus_until_complete(check_exit_time=1.0, restart=False)

    # Make offspring topologies
    if restart:
        _topo_offspring = np.genfromtxt('topo_offspring_' + str(gen) + '.csv', delimiter=',', dtype=int)
        _topo_offspring = _topo_offspring.reshape((params.end_pop, params.lx, params.ly, params.lz))
    else:
        _topo_offspring = generate_offspring(topo_parent=topo_parent, gen=gen, end_pop=params.end_pop,
                                             timeout=params.timeout, mutation_rate=params.mutation_rate,
                                             lx=params.lx, ly=params.ly, lz=params.lz)

    # Make abaqus work
    wait_for_abaqus_until_complete(check_exit_time=1.0, restart=restart, w=gen, offspring=_topo_offspring)

    # Import parent outputs of current generation from abaqus
    topo_offspring, result_offspring = offspring_import(gen_num=gen)
    fitness_values_parent = evaluate_fitness_values(topo=topo_parent, result=result_parent, params=params)
    fitness_values_offspring = evaluate_fitness_values(topo=topo_offspring, result=result_offspring, params=params)
    fitness_values_parent_and_offspring = np.vstack((fitness_values_parent, fitness_values_offspring))

    # Topologies of parent of next generation will be selected by pareto fronts criterion
    _, next_generations = selection(all_topologies=np.vstack((topo_parent, topo_offspring)),
                                    all_fitness_values=fitness_values_parent_and_offspring,
                                    population_size=params.end_pop)

    # The selected parent topologies are now exported as a CSV file
    parent_export(topo_parent=topo_parent, topo_offspring=topo_offspring,
                  result_parent=result_parent, result_offspring=result_offspring,
                  gen_num=gen, population_size=params.end_pop, next_generations=next_generations)

    # Next generations do not restart
    if restart:
        params.restart_pop = 0
    print('Generation:', gen)

    # Visualize
    if visualizer is not None:
        visualizer.visualize(params=params, w=gen, use_manual_rp=False, ref_x=0.0, ref_y=0.0)


def plot_previous_data(visualizer: Visualizer, use_manual_rp: bool, ref_x: float = 0.0, ref_y: float = 0.0) -> None:
    """
    Plot pareto fronts and hyper volumes of all worked generations.
    :param visualizer: Visualizer class
    :param use_manual_rp: If True, a reference point coordinate is fixed as your input point (ref_x, ref_y).
    If False, reference point will be calculated automatically.
    The reference point is used as a point for calculating hyper volume.
    :param ref_x: X coordinate of reference point for reference point if the use_manual_rp is True.
    :param ref_y: Y coordinate of reference point for reference point if the use_manual_rp is True.
    :return: Nothing
    """
    if os.path.isfile(PLOTTING_DATA_FILENAME):
        with open(PLOTTING_DATA_FILENAME, mode='rb') as f_read:
            read_data = pickle.load(f_read)
        for generation_num, (pareto_1_sorted, pareto_2_sorted) in read_data.items():
            visualizer.plot(gen_num=generation_num, pareto_1_sorted=pareto_1_sorted, pareto_2_sorted=pareto_2_sorted,
                            use_manual_rp=use_manual_rp, ref_x=ref_x, ref_y=ref_y)


if __name__ == '__main__':
    # Make an interface and receive parameters
    gui_process, parent_conn, child_conn = make_and_start_process(target=App, duplex=True, daemon=True)
    set_path, parameters = parent_conn.recv()
    parameters.post_initialize()
    os.chdir(set_path)
    remove_file(file_name=ABAQUS_ARGUMENTS_FILENAME)

    # load previous plotting data
    v = Visualizer(conn_to_gui=parent_conn)
    plot_previous_data(visualizer=v, use_manual_rp=False)

    # Open an abaqus process
    abaqus_process = open_abaqus(abaqus_script_name=parameters.abaqus_script_name, params=parameters,
                                 abaqus_execution_mode=parameters.abaqus_execution_mode, directory=set_path)

    # Start GA
    try:
        if parameters.mode == 'GA':
            if parameters.restart_pop == 0:
                for gen_num in range(parameters.ini_gen, parameters.end_gen + 1):
                    one_generation(gen=gen_num, restart=False, params=parameters, visualizer=v)
            else:
                one_generation(gen=parameters.ini_gen, restart=True, params=parameters, visualizer=v)
                for gen_num in range(parameters.ini_gen + 1, parameters.end_gen + 1):
                    one_generation(gen=gen_num, restart=False, params=parameters, visualizer=v)
        elif parameters.mode == 'Something':
            pass

        # Make abaqus exit itself
        with open(ABAQUS_PROCESS_END_FILENAME, mode='wb') as f_args_end:
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
        raise KeyboardInterrupt
