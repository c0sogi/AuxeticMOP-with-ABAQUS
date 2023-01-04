import pandas as pd
from dataclasses import dataclass, field
from multiprocessing import Process, Pipe
from GraphicUserInterface import App
from datetime import datetime
from time import sleep
from scipy.ndimage import gaussian_filter
from GeneticAlgorithm import generate_offspring, array_to_csv
from PostProcessing import *


@dataclass(kw_only=True)
class Parameters:
    abaqus_script_name: str
    abaqus_execution_mode: str
    mode: str
    evaluation_version: str
    restart_pop: int
    ini_pop: int
    end_pop: int
    ini_gen: int
    end_gen: int
    mutation_rate: float
    unit_l: float
    lx: int
    ly: int
    lz: int
    divide_number: int
    mesh_size: float
    dis_y: float
    material_modulus: float
    poissons_ratio: float
    density: float
    MaxRF22: float
    penalty_coefficient: float
    sigma: float
    threshold: float
    n_cpus: int
    n_gpus: int
    add_probability: float = 0.01
    timeout: float = 0.5
    created_at: str = field(default_factory=lambda: str(datetime.now()))

    def __post_init__(self):
        self.lx *= self.divide_number
        self.ly *= self.divide_number
        self.lz *= self.divide_number  # number of voxels after increasing resolution
        self.unit_l /= self.divide_number
        self.unit_l_half = self.unit_l * 0.5
        self.unit_lx_total = self.lx * self.unit_l
        self.unit_ly_total = self.ly * self.unit_l
        self.unit_lz_total = self.lz * self.unit_l
        self.mesh_size *= self.unit_l
        self.dis_y *= self.unit_ly_total  # boundary condition (displacement)
        self.MaxRF22 *= self.unit_lx_total * self.unit_lz_total * self.material_modulus  # 0.01 is strain

    def update(self, file_name):
        with open(file=file_name, mode='rb') as f:
            load = pickle.load(f)
        for key, value in load.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.__post_init__()


def make_and_start_process(target):
    conn_1, conn_2 = Pipe(duplex=True)
    process = Process(target=target, args=(conn_2,))
    process.start()
    return process, conn_1, conn_2


def load_pickle_and_set_parameters(file_name):
    with open(file=file_name, mode='rb') as f_read:
        return Parameters(**pickle.load(f_read))


def remove_file(file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)
        return True
    else:
        return False


def open_abaqus(abaqus_script_name, params, abaqus_execution_mode):
    with open('./PARAMS', mode='wb') as f_params:
        pickle.dump(params, f_params, protocol=2)
    print(f"========== Opening ABAQUS on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}! ==========")
    process = Process(target=os.system, args=(f'abaqus cae {abaqus_execution_mode}={abaqus_script_name}',), daemon=True)
    process.start()
    return process


def wait_for_abaqus_to_complete(check_exit_time, restart, w, offspring):
    args = {
        'restart': restart,
        'w': w,
        'offspring': offspring,
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


def read_numpy_from_csv(file_name, w, from_type, to_type):
    return pd.read_csv(f'./{file_name}_{w}.csv', header=None, dtype=from_type).to_numpy(
        dtype=to_type)  # data type transform


def array_divide(topo, lx, ly, lz, divide_number, ini_pop, end_pop):
    lxx = lx / divide_number
    lyy = ly / divide_number
    lzz = lz / divide_number
    topo = topo.reshape((end_pop, lzz, lyy, lxx))
    topo_divided = np.zeros((end_pop, lz, ly, lx))
    for q in range(ini_pop, end_pop + 1):
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    topo_divided[q - 1][k][j][i] = topo[q - 1][k // divide_number][j // divide_number][
                        i // divide_number]
    return topo_divided


def filter_process(topo_divided, sigma, threshold, lx, ly, lz, ini_pop, end_pop):
    topo_filtered = np.zeros((end_pop, lz, ly, lx))
    for q in range(ini_pop, end_pop):
        topo_divided2 = gaussian_filter(topo_divided[q - 1], sigma=sigma)
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    if topo_divided2[k][j][i] >= threshold:
                        topo_divided2[k][j][i] = 1
                    else:
                        topo_divided2[k][j][i] = 0
        topo_filtered[q - 1] = topo_divided2
    return topo_filtered


def offspring_import(w, mode):
    if mode == 'Random':
        topo_1 = np.genfromtxt('topo_parent_' + str(w) + '.csv', delimiter=',', dtype=int)

    else:
        topo_1 = np.genfromtxt('topo_offspring_' + str(w) + '.csv', delimiter=',', dtype=int)

    reslt_1 = np.genfromtxt('Output_offspring_' + str(w) + '.csv', delimiter=',', dtype=np.float32)
    return topo_1, reslt_1


def crowding_calculation(fitness_values):
    pop_size = len(fitness_values[:, 0])
    fitness_value_number = len(fitness_values[0, :])
    matrix_for_crowding = np.zeros((pop_size, fitness_value_number))
    normalize_fitness_values = (fitness_values - fitness_values.min(0)) / fitness_values.ptp(0)

    # normalize the fitness values
    for i in range(fitness_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1  # extreme point has the max crowding distance
        crowding_results[pop_size - 1] = 1  # extreme point has the max crowding distance
        sorting_normalize_fitness_values = np.sort(normalize_fitness_values[:, i])
        sorting_normalized_values_index = np.argsort(normalize_fitness_values[:, i])

        # crowding distance calculation
        crowding_results[1:pop_size - 1] = (
                sorting_normalize_fitness_values[2:pop_size] - sorting_normalize_fitness_values[0:pop_size - 2])
        re_sorting = np.argsort(sorting_normalized_values_index)  # re_sorting to the orginal order
        matrix_for_crowding[:, i] = crowding_results[re_sorting]

    crowding_distance = np.sum(matrix_for_crowding, axis=1)  # crowding distance of each solution
    return crowding_distance


def remove_using_crowding(fitness_values, number_solutions_needed):
    rn = np.random  # addition
    pop_index = np.arange(fitness_values.shape[0])
    crowding_distance = crowding_calculation(fitness_values)
    selected_pop_index = np.zeros(number_solutions_needed)
    selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))

    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            # solution 1 is better than solution 2
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, solution_1, axis=0)  # remove the selected solution
            fitness_values = np.delete(fitness_values, (solution_1),
                                       axis=0)  # remove the fitness of the selected solution
            crowding_distance = np.delete(crowding_distance, (solution_1),
                                          axis=0)  # remove the related crowding distance

        else:
            # solution 2 is better than solution 1
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, solution_2, axis=0)
            fitness_values = np.delete(fitness_values, solution_2, axis=0)
            crowding_distance = np.delete(crowding_distance, solution_2, axis=0)

    selected_pop_index = np.asarray(selected_pop_index, dtype=int)  # Convert the data to integer
    return selected_pop_index


def selection(pop, fitness_values, pop_size):
    pop_index_0 = np.arange(pop.shape[0])
    pop_index = np.arange(pop.shape[0])
    pareto_front_index = []

    while len(pareto_front_index) < pop_size:
        new_pareto_front = pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
        total_pareto_size = len(pareto_front_index) + len(new_pareto_front)

        # check the size of pareto front, if larger than pop_size, remove some solutions using crowding criterion
        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(pareto_front_index)
            selected_solutions = (remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed))
            new_pareto_front = new_pareto_front[selected_solutions]

        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front))  # add to pareto
        remaining_index = set(pop_index) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))

    selected_pop_index = pareto_front_index.astype(int)
    selected_pop_topo = pop[pareto_front_index.astype(int)]
    return selected_pop_topo, selected_pop_index


def parent_export(w, next_generations, end_pop, results, results_1, topologies, topologies_1):
    for i in next_generations:
        if i < end_pop:
            array_to_csv(f'Output_parent_{w + 1}.csv', results[i], dtype=np.float32, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topologies[i], dtype=int, mode='a', save_as_int=True)

        if i >= end_pop:
            array_to_csv(f'Output_parent_{w + 1}.csv', results_1[i - end_pop], dtype=np.float32, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topologies_1[i - end_pop], dtype=int, mode='a', save_as_int=True)


def one_generation(w, restart, set_path, parent_conn, parameters):
    topologies, results = parent_import(w=w, restart_pop=parameters.restart_pop)
    offspring = generate_offspring(topologies=topologies, w=w, end_pop=parameters.end_pop,
                                   timeout=parameters.timeout, mutation_rate=parameters.mutation_rate,
                                   add_probability=parameters.add_probability,
                                   lx=parameters.lx, ly=parameters.ly, lz=parameters.lz)
    wait_for_abaqus_to_complete(check_exit_time=1, restart=restart, w=w, offspring=offspring)

    topologies_1, results_1 = offspring_import(w=w, mode=parameters.mode)
    fitness_values = evaluation(topo=topologies, topo_1=topologies_1, reslt=results, reslt_1=results_1,
                                lx=parameters.lx, ly=parameters.ly, lz=parameters.lz, max_rf22=parameters.MaxRF22,
                                evaluation_version=parameters.evaluation_version, q=parameters.end_pop,
                                penalty_coefficient=parameters.penalty_coefficient)

    pop, next_generations = selection(pop=np.append(topologies, topologies_1, axis=0), fitness_values=fitness_values,
                                      pop_size=parameters.end_pop)
    parent_export(topologies=topologies, topologies_1=topologies_1, results=results, results_1=results_1,
                  w=w, end_pop=parameters.end_pop, next_generations=next_generations)
    if restart:
        parameters.restart_pop = 0
    print('iteration:', w)
    visualize(w=w, restart_pop=parameters.restart_pop, lx=parameters.lx, ly=parameters.ly, lz=parameters.lz,
              penalty_coefficient=parameters.penalty_coefficient, evaluation_version=parameters.evaluation_version,
              max_rf22=parameters.MaxRF22, directory=set_path, end_pop=parameters.end_pop, parent_conn=parent_conn)


if __name__ == '__main__':
    # Make an interface
    app, parent_conn, child_conn = make_and_start_process(target=App)
    set_path = parent_conn.recv()

    # Changed current dir to set_path and load parameters
    os.chdir(set_path)
    parameters = load_pickle_and_set_parameters(file_name='PARAMS_MAIN')
    remove_file(file_name='args')

    # Open a abaqus process
    abaqus_process = open_abaqus(abaqus_script_name=parameters.abaqus_script_name, params=parameters,
                                 abaqus_execution_mode=parameters.abaqus_execution_mode)

    # Start working
    try:
        if parameters.mode == 'GA':
            if parameters.restart_pop == 0:
                for gen_idx in range(parameters.ini_gen, parameters.end_gen + 1):
                    one_generation(w=gen_idx, restart=False, parameters=parameters, set_path=set_path,
                                   parent_conn=parent_conn)
            else:
                one_generation(w=parameters.ini_gen, restart=True, set_path=set_path, parameters=parameters,
                               parent_conn=parent_conn)
                for gen_idx in range(parameters.ini_gen + 1, parameters.end_gen + 1):
                    one_generation(w=gen_idx, restart=False, parameters=parameters, set_path=set_path,
                                   parent_conn=parent_conn)
        elif parameters.mode == 'Something':
            pass
        # Make abaqus exits itself
        with open('args_end', mode='wb') as f_args_end:
            pickle.dump('end', f_args_end, protocol=2)
    except Exception as error_message:
        # An error message
        print('An error in main function occurred: ', error_message)
    finally:
        # Clear processes
        app.join()
        abaqus_process.join()
        app.close()
        parent_conn.close()
        child_conn.close()
        abaqus_process.close()