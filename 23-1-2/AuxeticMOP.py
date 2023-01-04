import os
import pickle
from scipy.ndimage import gaussian_filter
from datetime import datetime
from time import sleep
import pandas as pd
import threading
import GraphicUserInterface as GUI
from GeneticAlgorithm import generate_offspring, array_to_csv
from PostProcessing import *

app = GUI.App()
while True:
    if app.message is not None:
        setPath = app.message
        print(setPath)
        break
    else:
        pass
    sleep(1)

print('setPath: ', setPath)
os.chdir(setPath)
with open('./PARAMS_MAIN', mode='rb') as f:
    PARAMS_MAIN = pickle.load(f)

# Fake variable declaration, No effect!
abaqus_script_name = 'abaqus_script_new.py'
abaqus_execution_mode = 'noGUI'
mode = 'GA'
evaluation_version = 'ver3'
restart_pop = 0
ini_pop = 1
end_pop = 100
ini_gen = 1
end_gen = 50
mutation_rate = 10
unit_l = 3
lx = 0
ly = 10
lz = 10
divide_number = 1
mesh_size = 0.25
dis_y = -0.005
material_modulus = 1100
poissons_ratio = 0.4
density = 1
MaxRF22 = 0.01
penalty_coefficient = 0.1
sigma = 1
threshold = 0.5
n_cpus = 8
n_gpus = 0

add_probability = 0.01
timeout = 0.5

locals().update(PARAMS_MAIN)

lx = divide_number * lx
ly = divide_number * ly
lz = divide_number * lz  # number of voxels after increasing resolution
unit_l = unit_l / divide_number
unit_l_half = unit_l * 0.5
unit_lx_total = lx * unit_l
unit_ly_total = ly * unit_l
unit_lz_total = lz * unit_l
mesh_size *= unit_l
dis_y *= unit_ly_total  # boundary condition (displacement)
MaxRF22 *= unit_lx_total * unit_lz_total * material_modulus  # 0.01 is strain

PARAMS = {
    'setPath': setPath,
    'mode': mode,
    'restart_pop': restart_pop,
    'ini_pop': ini_pop,
    'end_pop': end_pop,
    'divide_number': divide_number,
    'unit_l': unit_l,
    'lx': lx,
    'ly': ly,
    'lz': lz,
    'mesh_size': mesh_size,
    'dis_y': dis_y,
    'density': density,
    'material_modulus': material_modulus,
    'poissons_ratio': poissons_ratio,
    'MAXRF22': MaxRF22,
    'penalty_coefficient': penalty_coefficient,
    'n_cpus': n_cpus,
    'n_gpus': n_gpus
}
print(PARAMS)


def save_variable_for_debugging(debug_code, w, debug_variable):
    debug_code = debug_code
    gen = w
    with open(f'./debug_gen_{gen}_code_{debug_code}', mode='wb') as f_debug:
        pickle.dump(debug_variable, f_debug)
    print(f'Debug code {debug_code} in generation {gen} done!')


def run_abaqus_script_without_gui(abaqus_script_name, params, abaqus_execution_mode):
    # Dumping PARAMS
    with open('./PARAMS', mode='wb') as f_params:
        pickle.dump(params, f_params, protocol=2)
    # Start of abaqus job
    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    print(f'>>>>> Opening ABAQUS on {now}! <<<<<')
    th = threading.Thread(target=os.system, args=[f'abaqus cae {abaqus_execution_mode}={abaqus_script_name}'],
                          daemon=True)
    print('Opening ABAQUS...')
    return th


def wait_for_abaqus_job_done(check_exit_time):
    print('Waiting for abaqus')
    while True:
        sleep(check_exit_time)
        if os.path.isfile('./args'):
            print('.', end='')
            continue
        else:
            print()
            break
    # End of abaqus job
    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    print(f'===== An abaqus job done on {now}!')


def read_numpy_from_csv(file_name, w, from_type, to_type):
    return pd.read_csv(f'./{file_name}_{w}.csv', header=None, dtype=from_type).to_numpy(
        dtype=to_type)  # data type transform


def array_divide(topo):
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


def filter_process(topo_divided, sigma, threshold):
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


def offspring_import(w):
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


def parent_export(w):
    for i in next_generations:
        if i < end_pop:
            array_to_csv(f'Output_parent_{w + 1}.csv', results[i], dtype=np.float32, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topologies[i], dtype=int, mode='a')

        if i >= end_pop:
            array_to_csv(f'Output_parent_{w + 1}.csv', results_1[i - end_pop], dtype=np.float32, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topologies[i - end_pop], dtype=int, mode='a')


if __name__ == '__main__':
    # Open ABAQUS until the main function of this script ends
    th_abaqus = run_abaqus_script_without_gui(abaqus_script_name=abaqus_script_name, params=PARAMS,
                                              abaqus_execution_mode=abaqus_execution_mode)  # Define an abaqus thread
    th_abaqus.daemon = True
    th_abaqus.start()  # start abaqus thread
    # For-loop of abaqus jobs
    if mode == 'GA':
        if restart_pop == 0:
            for w in range(ini_gen, end_gen + 1):
                topologies, results = parent_import(w=w, restart_pop=restart_pop)
                offspring = generate_offspring(topologies=topologies, w=w, end_pop=end_pop, timeout=timeout,
                                               mutation_rate=mutation_rate, add_probability=add_probability,
                                               lx=lx, ly=ly, lz=lz)

                # ********** start of an ABAQUS job **********
                args = {
                    'restart': False,
                    'w': w,
                    'offspring': offspring,
                }
                with open('./args', mode='wb') as f_args:
                    pickle.dump(args, f_args, protocol=2)
                wait_for_abaqus_job_done(check_exit_time=1)
                # ********** end of an ABAQUS job **********

                topologies_1, results_1 = offspring_import(w)  # data import
                fitness_values = evaluation(topo=topologies, topo_1=topologies_1, reslt=results, reslt_1=results_1,
                                            lx=lx, ly=ly, lz=lz, evaluation_version=evaluation_version,
                                            max_rf22=MaxRF22,
                                            q=end_pop,
                                            penalty_coefficient=penalty_coefficient)  # calculate fitness values
                # save_variable_for_debugging(debug_code=1, w=w, debug_variable=[topologies_1, results_1, fitness_values])

                pop, next_generations = selection(np.append(topologies, topologies_1, axis=0), fitness_values,
                                                  end_pop)  # selection (index)
                # save_variable_for_debugging(debug_code=3, w=w, debug_variable=[pop, next_generations])

                parent_export(w)
                print('iteration:', w)
                visualize(w=w, restart_pop=restart_pop, lx=lx, ly=ly, lz=lz, end_gen=end_gen,
                          penalty_coefficient=penalty_coefficient, evaluation_version=evaluation_version,
                          max_rf22=MaxRF22, directory=setPath, end_pop=end_pop, app=app)

        else:
            w = ini_gen
            topologies, results, offspring = parent_import(w=w, restart_pop=restart_pop)
            offspring = offspring.reshape((end_pop, lx, ly, lz))
            # ********** start of an ABAQUS job **********
            args = {
                'restart': True,
                'w': w,
                'offspring': offspring,
            }
            with open('./args', mode='wb') as f_args:
                pickle.dump(args, f_args, protocol=2)
            wait_for_abaqus_job_done(check_exit_time=1)
            # ********** end of an ABAQUS job **********

            topologies_1, results_1 = offspring_import(w)  # data import
            fitness_values = evaluation(topo=topologies, topo_1=topologies_1, reslt=results, reslt_1=results_1,
                                        lx=lx, ly=ly, lz=lz, evaluation_version=evaluation_version, max_rf22=MaxRF22,
                                        q=end_pop, penalty_coefficient=penalty_coefficient)  # calculate fitness values
            pop, next_generations = selection(np.append(topologies, topologies_1, axis=0), fitness_values,
                                              end_pop)  # selection (index)
            parent_export(w)
            print('iteration:', w)
            restart_pop = 0

            for w in range(ini_gen + 1, end_gen + 1):
                topologies, results = parent_import(w=w, restart_pop=restart_pop)
                offspring = generate_offspring(topologies=topologies_1, w=w, end_pop=end_pop, timeout=timeout,
                                               mutation_rate=mutation_rate, add_probability=add_probability,
                                               lx=lx, ly=ly, lz=lz)

                # ********** start of an ABAQUS job **********
                args = {
                    'restart': False,
                    'w': w,
                    'offspring': offspring,
                }
                with open('./args', mode='wb') as f_args:
                    pickle.dump(args, f_args, protocol=2)
                wait_for_abaqus_job_done(check_exit_time=1)
                # ********** end of an ABAQUS job **********

                topologies_1, results_1 = offspring_import(w)  # data import
                fitness_values = evaluation(topo=topologies, topo_1=topologies_1, reslt=results, reslt_1=results_1,
                                            lx=lx, ly=ly, lz=lz, evaluation_version=evaluation_version,
                                            max_rf22=MaxRF22,
                                            q=end_pop,
                                            penalty_coefficient=penalty_coefficient)  # calculate fitness values
                # save_variable_for_debugging(debug_code=4, w=w, debug_variable=[topologies_1, results_1, fitness_values])

                pop, next_generations = selection(np.append(topologies, topologies_1, axis=0), fitness_values,
                                                  end_pop)  # selection (index)
                # save_variable_for_debugging(debug_code=6, w=w, debug_variable=[pop, next_generations])

                parent_export(w)
                print('iteration:', w)
                visualize(w=w, restart_pop=restart_pop, lx=lx, ly=ly, lz=lz, end_gen=end_gen,
                          penalty_coefficient=penalty_coefficient, evaluation_version=evaluation_version,
                          max_rf22=MaxRF22, directory=setPath, end_pop=end_pop, app=app)

    with open('./args_end', mode='wb') as f_args_end:
        pickle.dump('end', f_args_end, protocol=2)
    th_abaqus.join()  # this python script exits only when abaqus process is closed manually.
