import numpy as np
import csv
import matplotlib.pyplot as plt
from GeneticAlgorithm import parent_import


def evaluation(topo, topo_1, reslt, reslt_1, q, lx, ly, lz, evaluation_version, penalty_coefficient, max_rf22):
    fitness_values = np.zeros((2 * q, 2))
    k = penalty_coefficient
    if evaluation_version == 'ver1':
        for i in range(q):
            dis11 = reslt[i][0]
            dis22 = reslt[i][1]
            RF22 = reslt[i][4]
            fit_val1 = (RF22 / max_rf22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

        for i in range(q, 2 * q):
            dis11 = reslt_1[i - q][0]
            dis22 = reslt_1[i - q][1]
            RF22 = reslt_1[i - q][4]
            fit_val1 = (RF22 / max_rf22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver2':
        for i in range(q):
            RF22 = reslt[i][4]
            fit_val1 = np.sum(topo[i]) / (lx * ly * lz)
            fitness_values[i][0] = fit_val1
            fit_val2 = RF22 / max_rf22
            fitness_values[i][1] = fit_val2

        for i in range(q, 2 * q):
            RF22 = reslt_1[i - q][4]
            fit_val1 = np.sum(topo_1[i - q]) / (lx * ly * lz)
            fitness_values[i][0] = fit_val1
            fit_val2 = RF22 / max_rf22
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver3':
        for i in range(q):
            dis11 = reslt[i][0]
            dis22 = reslt[i][1]
            dis33 = reslt[i][2]
            fit_val1 = - (dis11 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis33 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

        for i in range(q, 2 * q):
            print('reslt1', reslt_1, reslt_1.shape)
            dis11 = reslt_1[i - q][0]
            dis22 = reslt_1[i - q][1]
            dis33 = reslt_1[i - q][2]
            fit_val1 = - (dis11 / dis22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis33 / dis22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver4':
        for i in range(q):
            fit_val1 = reslt[i][9]
            fitness_values[i][0] = fit_val1
            fit_val2 = np.sum(topo[i]) / (lx * ly * lz)
            fitness_values[i][1] = fit_val2
        for i in range(q, 2 * q):
            fit_val1 = reslt_1[i - q][9]
            fitness_values[i][0] = fit_val1
            fit_val2 = np.sum(topo_1[i - q]) / (lx * ly * lz)
            fitness_values[i][1] = fit_val2
    return fitness_values


def evaluation2(topo2, reslt2, lx, ly, lz, penalty_coefficient, evaluation_version, max_rf22):
    fitness_values = np.zeros((topo2.shape[0], 2))
    k = penalty_coefficient
    if evaluation_version == 'ver1':
        for i in range(topo2.shape[0]):
            dis11 = reslt2[i][0]
            dis22 = reslt2[i][1]
            RF22 = reslt2[i][4]
            fit_val1 = (RF22 / max_rf22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver2':
        for i in range(topo2.shape[0]):
            RF22 = reslt2[i][4]
            fit_val1 = np.sum(topo2[i]) / (lx * ly * lz)
            fitness_values[i][0] = fit_val1
            fit_val2 = RF22 / max_rf22
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver3':
        for i in range(topo2.shape[0]):
            dis11 = reslt2[i][0]
            dis22 = reslt2[i][1]
            dis33 = reslt2[i][2]
            fit_val1 = - (dis11 / dis22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis33 / dis22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver4':
        for i in range(topo2.shape[0]):
            fit_val1 = reslt2[i][9]
            fitness_values[i][0] = fit_val1
            fit_val2 = np.sum(topo2[i]) / (lx * ly * lz)
            fitness_values[i][1] = fit_val2

    return fitness_values


def max_fitval(w, restart_pop, lx, ly, lz, evaluation_version, penalty_coefficient, max_rf22):
    for i in range(1, w + 1):
        topo, result = parent_import(w=i, restart_pop=restart_pop)
        fitness_values = evaluation2(topo2=topo, reslt2=result, lx=lx, ly=ly, lz=lz,
                                     evaluation_version=evaluation_version,
                                     max_rf22=max_rf22, penalty_coefficient=penalty_coefficient)
        if i == 1:
            fit_max = np.max(fitness_values, axis=0)
            fit_min = np.min(fitness_values, axis=0)
        else:
            fit_max1 = np.max(fitness_values, axis=0)
            fit_min1 = np.min(fitness_values, axis=0)
            for i in range(2):
                if fit_max1[i] > fit_max[i]:
                    fit_max[i] = fit_max1[i]
                if fit_min1[i] < fit_min[i]:
                    fit_min[i] = fit_min1[i]
    return fit_max, fit_min


def fitval_sort(fv):
    fv_sp = np.split(fv, 2, axis=1)
    fv1 = fv_sp[0]
    fv2 = fv_sp[1]
    fv1 = np.squeeze(fv1)
    fv2 = np.squeeze(fv2)
    sort = fv1.argsort()
    fv1_sort = fv1[sort]
    fv2_sort = fv2[sort]
    return fv1_sort, fv2_sort, sort


def hypervolume_calculation(fv1_sort, fv2_sort, end_gen, restart_pop, lx, ly, lz, evaluation_version,
                            penalty_coefficient, max_rf22):
    fit_max, fit_min = max_fitval(w=end_gen, restart_pop=restart_pop, lx=lx, ly=ly, lz=lz,
                                  evaluation_version=evaluation_version, penalty_coefficient=penalty_coefficient,
                                  max_rf22=max_rf22)
    rp_max1 = fit_max[0] + 0.05 * (fit_max[0] - fit_min[0])
    rp_min1 = fit_min[0] - 0.05 * (fit_max[0] - fit_min[0])
    rp_max2 = fit_max[1] + 0.05 * (fit_max[1] - fit_min[1])
    rp_min2 = fit_min[1] - 0.05 * (fit_max[1] - fit_min[1])
    hv = 0
    area = (rp_max1 - rp_min1) * (rp_max2 - rp_min2)
    pareto_no = len(fv1_sort)
    for i in range(pareto_no):
        if i == 0:
            hv += (fv1_sort[i] - rp_min1) * (rp_max2 - fv2_sort[i]) + (rp_max1 - fv1_sort[pareto_no - 1 - i]) * (
                    rp_max2 - fv2_sort[pareto_no - 1 - i])
        else:
            hv += ((fv1_sort[i] - fv1_sort[i - 1]) * (rp_max2 - fv2_sort[i]) + (
                    fv1_sort[pareto_no - i] - fv1_sort[pareto_no - 1 - i]) * (
                           rp_max2 - fv2_sort[pareto_no - 1 - i])) * 0.5
    normalized_hv = hv / area
    return normalized_hv, hv


def find_model_output(w, q, directory, end_pop, end_gen):
    no_of_datafile = end_gen
    no_of_lines = end_pop
    if directory[-1] != '/' or directory[-1] != '\\':
        directory += '/'

    f = open(directory + 'topo_parent_%d.csv' % w, 'r')
    rd = csv.reader(f)
    topo = [i for i in rd]
    topo = np.array(topo)
    topo = topo.astype(np.float32)
    gen_num_eng = 0
    pop_num_eng = 0
    f = open(directory + 'Output_parent_%d.csv' % w, 'r')
    rd = csv.reader(f)
    Eng = [i for i in rd]
    Eng = np.array(Eng)
    Eng = Eng.astype(np.float32)

    for j in range(no_of_datafile + 1):
        break1 = 0
        if j == 0:
            f = open(directory + 'topo_parent_%d.csv' % (j + 1), 'r')
            rd = csv.reader(f)
            topo1 = [i for i in rd]
            topo1 = np.array(topo1)
            topo1 = topo1.astype(np.float32)
            f = open(directory + 'Output_parent_%d.csv' % (j + 1), 'r')
            rd = csv.reader(f)
            Eng1 = [i for i in rd]
            Eng1 = np.array(Eng1)
            Eng1 = Eng1.astype(np.float32)
        else:
            f = open(directory + 'topo_offspring_%d.csv' % j, 'r')
            rd = csv.reader(f)
            topo1 = [i for i in rd]
            topo1 = np.array(topo1)
            topo1 = topo1.astype(np.float32)
            f = open(directory + 'Output_offspring_%d.csv' % j, 'r')
            rd = csv.reader(f)
            Eng1 = [i for i in rd]
            Eng1 = np.array(Eng1)
            Eng1 = Eng1.astype(np.float32)
        for k in range(no_of_lines):
            if np.array_equal(topo[q], topo1[k]) and np.less_equal(np.absolute(Eng[q] - Eng1[k]), 2E-07).all():
                gen_num_eng = j
                pop_num_eng = k + 1
                break1 = 1
                break
        if break1 == 1:
            break
    return gen_num_eng, pop_num_eng


def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)  # initially assume all solutions are in pareto front by using "1"

    for i in range(pop_size):
        for j in range(pop_size):
            if np.less_equal(fitness_values[j], fitness_values[i]).all() and np.less(fitness_values[j], fitness_values[i]).any():
                pareto_front[i] = 0  # i is not in pareto front because j dominates i
                break  # no more comparision is needed to find out which one is dominated

    return pop_index[pareto_front]


def visualize(w, restart_pop, lx, ly, lz, end_gen, penalty_coefficient, evaluation_version, max_rf22, directory, end_pop):
    topo_p, reslt_p = parent_import(w + 1, restart_pop=restart_pop)
    fitness_values = evaluation2(topo2=topo_p, reslt2=reslt_p, lx=lx, ly=ly, lz=lz,
                                 penalty_coefficient=penalty_coefficient, evaluation_version=evaluation_version,
                                 max_rf22=max_rf22)
    index = np.arange(topo_p.shape[0]).astype(int)
    pareto_front_index = pareto_front_finding(fitness_values, index)
    index = reslt_p[pareto_front_index, :]  # correction: index >> reslt_p
    fitness_values_pareto = fitness_values[pareto_front_index]
    fv1_sort, fv2_sort, sort = fitval_sort(fitness_values_pareto)

    normalized_hypervolume, hypervolume = hypervolume_calculation(fv1_sort=fv1_sort, fv2_sort=fv2_sort, end_gen=end_gen,
                                                                  restart_pop=restart_pop,
                                                                  lx=lx, ly=ly, lz=lz,
                                                                  evaluation_version=evaluation_version,
                                                                  penalty_coefficient=penalty_coefficient,
                                                                  max_rf22=max_rf22)
    print('\n')
    print("iteration: %d" % w)
    print("_________________")
    print("Optimal solutions:")
    print("       x1               x2                 x3")
    print(index)  # show optimal solutions
    print("______________")
    print("Fitness values:")
    # print("  objective 1    objective 2")
    print("          Model              objective 1      objective 2      Job")
    print("            |                     |                |            |")
    for q in range(len(index)):
        gen_num, pop_num = find_model_output(w=w+1, q=sort[q], directory=directory, end_pop=end_pop, end_gen=end_gen)
        print("%dth generation %dth pop   [%E   %E]   Job-%d-%d" % (
            w + 1, q + 1, fitness_values[q, 0], fitness_values[q, 1], gen_num, pop_num))
    # print(fitness_values)
    ##pareto_front
    fit_max, fit_min = max_fitval(w=end_gen, restart_pop=restart_pop, lx=lx, ly=ly, lz=lz,
                                  evaluation_version=evaluation_version, penalty_coefficient=penalty_coefficient,
                                  max_rf22=max_rf22)
    rp_max1 = fit_max[0] + 0.05 * (fit_max[0] - fit_min[0])
    rp_min1 = fit_min[0] - 0.05 * (fit_max[0] - fit_min[0])
    rp_max2 = fit_max[1] + 0.05 * (fit_max[1] - fit_min[1])
    rp_min2 = fit_min[1] - 0.05 * (fit_max[1] - fit_min[1])
    plt.figure(1)
    plt.plot(fv1_sort, fv2_sort, marker='o', color='#2ca02c')
    plt.xlabel('Objective function 1')
    plt.ylabel('Objective function 2')
    plt.axis((rp_min1, rp_max1, rp_min2, rp_max2))
    ##hypervolume
    print(normalized_hypervolume, hypervolume)
    plt.figure(2)
    plt.scatter(w, normalized_hypervolume)
    plt.xlabel('Iteration')
    plt.ylabel('Hypervolume')
    plt.axis((0, end_gen + 1, 0, 1))  # plt.axis((xmin, xmax, ymin, ymax))
    plt.pause(0.1)
    print(sort)
