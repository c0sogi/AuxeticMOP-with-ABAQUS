import os
import pickle
import numpy as np
from time import sleep
from datetime import datetime
import multiprocessing as mp
from multiprocessing import connection
from dataclasses import asdict
from auxeticmop import generate_offspring, random_parent_generation
from auxeticmop import App, Visualizer
from auxeticmop import evaluate_all_fitness_values, selection
from auxeticmop import load_pickled_dict_data, dump_pickled_dict_data
from auxeticmop import Server
from auxeticmop.ParameterDefinitions import Parameters, fitness_definitions


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


def start_abaqus_cae(script_name: str, option: str) -> mp.Process:
    """
    Open an abaqus CAE process
    :param script_name: Name of python script file for abaqus. For example, ABQ.py.
    :param option: 'noGUI' for abaqus non-gui mode, 'script' for abaqus gui mode.
    :return: Abaqus process.
    """
    print(f"========== Opening ABAQUS on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}! ==========")
    process = mp.Process(target=os.system, args=(f'abaqus cae {option}={script_name}',), daemon=True)
    process.start()  # Start abaqus
    return process


def wait_for_abaqus(restart: bool, topologies_file_name: str, params_dict: dict, topologies_key: str,
                    server: Server) -> None:
    """
    Hold main process until one generation of abaqus job is done.
    :param restart: Restarting evolution from unfinished generation previously done.
    :param topologies_file_name: File name of topologies
    :param topologies_key: Key of topology dict to analyze. e.g., 'offspring', 'parent'
    :param params_dict: Parameters converted as dictionary
    :param server: A server to ABAQUS
    :return: Nothing
    """
    json_data = params_dict.copy()
    json_data.update({
        'restart': restart,
        'topologies_file_name': topologies_file_name,
        'topologies_key': topologies_key,
        'exit_abaqus': False
    })
    server.send(client_socket=server.connected_clients[-1], data=json_data)
    print('Waiting for abaqus ...........')
    message_from_client = server.recv()
    print(message_from_client)
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
    try:
        topo_parent = load_pickled_dict_data(f'Topologies_{gen}')['parent']
    except FileNotFoundError or KeyError:
        topo_parent = random_parent_generation(gen=gen, density=0.5, params=params, show_parent=False)
    try:
        result_parent = load_pickled_dict_data(f'FieldOutput_{gen}')
    except FileNotFoundError:
        wait_for_abaqus(restart=False, topologies_file_name='Topologies_1', params_dict=asdict(params),
                        server=server_to_abaqus, topologies_key='parent')
        result_parent = load_pickled_dict_data('FieldOutput_offspring_1')
        remove_file('FieldOutput_offspring_1')
        with open('FieldOutput_1', mode='wb') as f:
            pickle.dump(result_parent, f, protocol=2)

    # Make offspring topologies
    if restart:
        topo_offspring = load_pickled_dict_data(f'Topologies_{gen}')['offspring']
    else:
        topo_offspring = generate_offspring(topo_parent=topo_parent, gen=gen, end_pop=params.end_pop,
                                            timeout=params.timeout, mutation_rate=params.mutation_rate,
                                            lx=params.lx, ly=params.ly, lz=params.lz)

    # Make abaqus work
    wait_for_abaqus(restart=False, topologies_file_name=f'Topologies_{gen}', params_dict=asdict(params),
                    server=server_to_abaqus, topologies_key='offspring')

    # Import parent outputs of current generation from abaqus
    result_offspring = load_pickled_dict_data(f'FieldOutput_offspring_{gen}')
    assert len(topo_parent) == len(result_parent) == len(topo_offspring) == len(result_offspring)
    all_topologies = np.vstack((topo_parent, topo_offspring))
    # This dict union using pipe operator is allowed only for Python version >= 3.9
    all_results = dict(result_parent | {key + len(result_parent): value for key, value in result_offspring.items()})
    all_fitness_values = evaluate_all_fitness_values(fitness_definitions=fitness_definitions,
                                                     params_dict=asdict(params),
                                                     results=all_results, topologies=all_topologies)

    # Topologies of parent of next generation will be selected by pareto fronts criterion
    pareto_indices = selection(all_fitness_values=all_fitness_values, selected_size=params.end_pop)
    selected_topologies = all_topologies[pareto_indices]
    selected_results = {entity_num: all_results[pareto_idx + 1]
                        for entity_num, pareto_idx in enumerate(pareto_indices, start=1)}
    dump_pickled_dict_data(f'Topologies_{gen + 1}', key='parent', to_dump=selected_topologies, mode='w')
    with open(f'FieldOutput_{gen + 1}', mode='wb') as f:
        pickle.dump(selected_results, f, protocol=2)

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
    if os.path.isfile('_plotting_data_'):
        with open('_plotting_data_', mode='rb') as f_read:
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

    # load previous plotting data
    v = Visualizer(conn_to_gui=parent_conn)
    # plot_previous_data(visualizer=v, use_manual_rp=False)

    # Open an abaqus process and socket server
    server_to_abaqus = Server(host='localhost', port=12345, option='json', run_nonblocking=True)
    abaqus_process = start_abaqus_cae(script_name=parameters.abaqus_script, option=parameters.abaqus_mode)
    while len(server_to_abaqus.connected_clients) == 0:
        print('Connecting socket to ABAQUS ...')
        sleep(1.0)

    # Start GA
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
    server_to_abaqus.send(client_socket=server_to_abaqus.connected_clients[-1], data={'exit_abaqus': True})
