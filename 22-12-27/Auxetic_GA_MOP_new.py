import numpy as np
from collections import deque
import itertools
import random
import math
from itertools import combinations_with_replacement
import os
import pickle
import matplotlib.pyplot as plt #correction: add
from scipy.ndimage import gaussian_filter
from datetime import datetime
from time import sleep
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import threading


## Path options
setPath = r'C:\Users\user\Desktop\shshsh2\pycharm\data'  # The folder path of abaqus_scripts.py & csv files
abaqus_script_name = 'abaqus_scripts_new.py'
# default_files_in_directory = (abaqus_script_name, 'Output_parent_1.csv', 'topo_parent_1.csv')
# delete_files_except_default_files_when_starting = True
# if delete_files_except_default_files_when_starting:
#     for file in os.listdir():
#         os.remove(file) if file not in default_files_in_directory else None
abaqus_execution_mode = 'script' # 'noGUI' for cui mode or 'script' for gui mode


## Global variables
mode = 'GA'  # GA or Random or Gaussian or None
model = 'original'
evaluation_version = 'ver2'
# ver1: stiffness(RF22) + mass / poissons_ratio12
# ver2: mass / stiffness(RF22)
# ver3: poissons_ratio12 + mass / poissons_ratio13 + mass
# ver4: maximum von-mises stress / mass

# GA parameter
restart_pop = 0  # for the restart, zero means inactivated
#
ini_pop = 1  # Fixed value as 1
end_pop = 100  # population number
ini_gen = 1  # starting generation
end_gen = 50
mutation_rate = 10  # probability of mutation (percentage)

# CAE model variables
unit_l = 3  # original voxel size
lx = 10
ly = 10
lz = 10  # original design space (number of voxels)
divide_number = 1  # increasing resolution(reduce voxel size), 1 means orinal , 10 means 1 voxel devided into 1/10^3 size
#
lx = divide_number * lx
ly = divide_number * ly
lz = divide_number * lz  # number of voxels after increasing resolution
unit_l = unit_l / divide_number
unit_l_half = unit_l * 0.5
unit_lx_total = lx * unit_l
unit_ly_total = ly * unit_l
unit_lz_total = lz * unit_l
#
mesh_size = 0.25 * unit_l  # mesh size compare to voxelsize
dis_y = -0.005 * unit_ly_total  # boundary condition (displacement)
material_modulus = 1100
poissons_ratio = 0.4
density = 1
MaxRF22 = 0.01 * unit_lx_total * unit_lz_total * material_modulus  # 0.01 is strain
#
penalty_coefficient = 0.1  # mass penalty coefficient k
sigma = 1
threshhold = 0.5  # gaussian filter parameter

# CAE computing parameter
n_cpus = 8  # number of cpu usage
n_gpus = 0

PARAMS = {
    'setPath': setPath,
    'mode': mode,
    'model': model,
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


def array_to_csv(path, arr, dtype, mode, save_as_int=False):
    if mode == 'a' and os.path.isfile(path):
        previous_arr = csv_to_array(path, dtype=dtype)
        # print('[array_to_csv] append shape: ', previous_arr.shape, arr.shape)
        arr = np.vstack((previous_arr, arr))
    fmt = '%i' if save_as_int else '%.18e'
    np.savetxt(path, arr, delimiter=',', fmt=fmt)


def csv_to_array(path, dtype):
    return np.genfromtxt(path, delimiter=',', dtype=dtype)


def save_variable_for_debugging(debug_code, w, debug_variable):
    debug_code = debug_code
    gen = w
    with open(f'./debug_gen_{gen}_code_{debug_code}', mode='wb') as f_debug:
        pickle.dump(debug_variable, f_debug)
    print(f'Debug code {debug_code} in generation {gen} done!')


def run_abaqus_script_without_gui(set_path, abaqus_script_name, params, abaqus_execution_mode):
    # Dumping PARAMS
    os.chdir(set_path)
    with open('./PARAMS', mode='wb') as f_params:
        pickle.dump(params, f_params, protocol=2)
    # Start of abaqus job
    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    print(f'>>>>> Opening ABAQUS on {now}! <<<<<')
    th = threading.Thread(target=os.system, args=[f'abaqus cae {abaqus_execution_mode}={abaqus_script_name}'])
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


# def delete_empty_rows_from_temp(file_name, w, from_type, to_type):
#     pd.read_csv(f'./{file_name}_{w}_temp.csv', header=None, dtype=from_type).to_csv(
#         f'./{file_name}_{w}.csv', header=False, index=False)
#     return pd.read_csv(f'./{file_name}_{w}.csv', header=None, dtype=from_type).to_numpy(
#         dtype=to_type)  # data type transform


def read_numpy_from_csv(file_name, w, from_type, to_type):
    return pd.read_csv(f'./{file_name}_{w}.csv', header=None, dtype=from_type).to_numpy(
        dtype=to_type)  # data type transform


# def topo_save(topo5):
#     topo7 = topo5.flatten()
#     e2 = open(directory + 'topology_random_temp.csv', 'a')
#     # e1 = open('topology.csv', 'a', encoding='utf-8', newline='')
#     wr = csv.writer(e2)
#     wr.writerow(topo7)
#     e2.close()


def bfs_alldirec(x, y, z, topology):
    topo = topology.copy()
    direction = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    visited = np.zeros((lz, ly, lx))
    queue = deque()
    queue.append([x, y, z])
    visited[z][y][x] = 1
    exit_x0 = False
    exit_x = False
    exit_y0 = False
    exit_y = False
    exit_z0 = False
    exit_z = False

    while queue:
        x, y, z = queue.popleft()
        for i in range(6):
            nx = x + direction[i][0]
            ny = y + direction[i][1]
            nz = z + direction[i][2]
            if nx in range(lx) and ny in range(ly) and nz in range(lz):
                if topo[nz][ny][nx] == 1 and visited[nz][ny][nx] == 0:
                    visited[nz][ny][nx] = 1
                    queue.append([nx, ny, nz])
                    # endpoint = 0
                    if nz == lz - 1:
                        exit_z = True
                    if nx == lx - 1:
                        exit_x = True
                    if ny == ly - 1:
                        exit_y = True
                    if nz == 0:
                        exit_z0 = True
                    if ny == 0:
                        exit_y0 = True
                    if nx == 0:
                        exit_x0 = True
    connect = [exit_x, exit_x0, exit_y, exit_y0, exit_z, exit_z0]

    if all(connect) == True:
        for x in range(lx):
            for y in range(ly):
                for z in range(lz):
                    if topo[z][y][x] == 1 and visited[z][y][x] == 0:
                        topo[z][y][x] = 0
        global topoend
        topoend = 1
        print('good')
        # print('topo_refined',len(np.where(topo<0)[0]))
        return topo
    else:
        print('Trapped!')


def bfs_back(x, y, z, topology):
    topo = topology.copy()
    direction = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]
    visited = np.zeros((lz, ly, lx))
    queue = deque()
    queue.append([x, y, z])
    visited[z][y][x] = 1
    queue2 = deque()
    y1 = y
    while queue:
        x, y, z = queue.popleft()
        for i in range(5):
            nx = x + direction[i][0]
            ny = y + direction[i][1]
            nz = z + direction[i][2]
            if nx in range(lx) and ny in range(ly) and nz in range(lz):
                if topo[nz][ny][nx] == 1 and visited[nz][ny][nx] == 0:
                    visited[nz][ny][nx] = 1
                    queue.append([nx, ny, nz])
        if y == y1 and topo[z][y1 - 1][x] == 1:
            queue2.append([x, y1 - 1, z])
    # print('bfs_back_queue_end')

    ly_num = np.where(visited[:, ly - 1, :] == 1)
    if len(ly_num[0]) != 0:
        return topo
    if len(ly_num[0]) == 0:
        topo - visited
        # print('bfsback_one_negative',len(np.where(topo<0)[0]))
    while queue2:
        queue = deque()
        visited = np.zeros((lz, ly, lx))
        x, y, z = queue2.popleft()
        y1 = y
        x1 = x
        z1 = z

        queue.append([x, y, z])
        if topo[z][y][x] == 1:
            while queue:
                x, y, z = queue.popleft()
                for i in range(5):
                    nx = x + direction[i][0]
                    ny = y + direction[i][1]
                    nz = z + direction[i][2]
                    if nx in range(lx) and ny in range(ly) and nz in range(lz):
                        if topo[nz][ny][nx] + visited[nz][ny][nx] == 0:
                            visited[nz][ny][nx] = 1
                            queue.append([nx, ny, nz])
            # print('bfs_back_queue2_queue_end')

        ly_num = np.where(visited[:, ly - 1, :] == 1)

        if len(ly_num[0]) != 0:
            break

        else:
            if topo[z1][y1 - 1][x1] == 1:
                queue2.append([x1, y1 - 1, z1])
            topo - visited
            # print('bfsback_two_negative',len(np.where(topo<0)[0]))
    # print('bfs_back_queue2_end')
    return topo


def bfs_ydirec(x, y, z, topology):
    topo = topology.copy()
    direction = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]
    visited = np.zeros((lz, ly, lx))
    queue = deque()
    queue_back = deque()
    queue.append([x, y, z])
    visited[z][y][x] = 1

    while queue:
        x, y, z = queue.popleft()
        path_count = 0
        for i in range(5):
            nx = x + direction[i][0]
            ny = y + direction[i][1]
            nz = z + direction[i][2]

            if nx in range(lx) and ny in range(ly) and nz in range(lz):
                if topo[nz][ny][nx] == 1 and visited[nz][ny][nx] == 0:
                    visited[nz][ny][nx] = 1
                    queue.append([nx, ny, nz])
                    path_count += 1

        if path_count == 0:
            if y == ly - 2:
                visited[z][y + 1][x] = 1  # adding voxel for connection
            if y < ly - 2:
                queue_back.append([x, y, z])

    # print('y_direc_first_queue_end')
    while queue_back:
        x, y, z = queue_back.popleft()
        visited = bfs_back(x, y, z, visited)
    # print('y_direc_queue_back_end')
    ly_num = np.where(visited[:, ly - 1, :] == 1)
    if len(ly_num[0]) == 0:
        visited = np.zeros((lz, ly, lx))
    return visited


def bfs_total(topo):
    global_visited = np.zeros((lz,ly,lx))
    topo_refined = np.zeros((lz,ly,lx))
    b = np.where(topo[:,0,:]==1)
    # print('first_topo_y0_number : ',len(b[0]))\
    for i in range(len(b[0])):
        global topoend
        topo_refined = bfs_alldirec(b[1][i],0,b[0][i],topo)
        if topoend ==1:
            break   
    if topoend != 1:
        return
    
    bb = np.where(topo_refined[:,0,:]==1)
        # print('first_ydirec_loop :', i)
    for i in range(len(bb[0])):
        if topo[b[0][i]][0][b[1][i]]==1 and global_visited[b[0][i]][0][b[1][i]]==0: 
            global_visited += bfs_ydirec(b[1][i],0,b[0][i],topo_refined)
    global_visited = np.where(global_visited>0,1,global_visited)
    # print('global_visited_negative',len(np.where(global_visited<0)[0]))
    topo_crs = design_const_one(global_visited)
    if topo_crs == 1:
        print('design is constrained')
        topoend = 0
        return 
    else: 
        print('design is verified')
        return global_visited


def array_divide(topo):
    lxx = lx / divide_number;
    lyy = ly / divide_number;
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


def filter_process(topo_divided, sigma, threshhold):
    topo_filtered = np.zeros((end_pop, lz, ly, lx))
    for q in range(ini_pop, end_pop):
        topo_divided2 = gaussian_filter(topo_divided[q - 1], sigma=sigma)
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    if topo_divided2[k][j][i] >= threshhold:
                        topo_divided2[k][j][i] = 1
                    else:
                        topo_divided2[k][j][i] = 0
        topo_filtered[q - 1] = topo_divided2
    return topo_filtered


def crossover(x, y, cutting):  # Crossover process;      correction: add input 'cutting'
    offspring1 = np.zeros(topo.shape[1])
    offspring2 = np.zeros(topo.shape[1])
    offspring1[0:cutting] = x[0:cutting]
    offspring1[cutting:] = y[cutting:]
    offspring2[0:cutting] = y[0:cutting]
    offspring2[cutting:] = x[cutting:]
    offsprings = np.vstack([offspring1, offspring2])
    return offsprings


def mutation(topologys, mutationrate):
    for j in range(2):
        for i in range(len(topologys[0])):
            p = np.random.randint(1, 101)
            if mutationrate > p:
                if topologys[j][i] == 1:
                    topologys[j][i] = 0
                elif topologys[j][i] == 0:
                    topologys[j][i] = 1
    return topologys


def design_const_one(topo):
    flag = 1
    direc = [1,-1]
    while flag:
        #a=shuffle range(a)
        flag=0
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    topcrs = 0
                    if topo[k][j][i] == 1:
                        for x in range(2):
                            for y in range(2):
                                if k+direc[x] in range(lz) and j+direc[y] in range(ly):
                                    if topo[k+direc[x]][j+direc[y]][i] == 1:
                                        if topo[k+direc[x]][j][i] == 0 and topo[k][j+direc[y]][i] == 0:
                                            # topo[k][j][i] = 0
                                            topcrs = 1
                                            flag = 1
                                            break

                                if k+direc[x] in range(lz) and i+direc[y] in range(lx):
                                    if topo[k+direc[x]][j][i+direc[y]] == 1:
                                        if topo[k+direc[x]][j][i] == 0 and topo[k][j][i+direc[y]] == 0:
                                            # topo[k][j][i] = 0
                                            topcrs = 1
                                            flag = 1
                                            break

                                if j + direc[x] in range(ly) and i + direc[y] in range(lx):
                                    if topo[k][j + direc[x]][i + direc[y]] == 1:
                                        if topo[k][j + direc[x]][i] == 0 and topo[k][j][i + direc[y]] == 0:
                                            # topo[k][j][i] = 0
                                            topcrs = 1
                                            flag = 1
                                            break

                                for z in range(2):
                                    if k+direc[x] in range(lz) and j+direc[y] in range(ly) and i+direc[z] in range(lx):
                                        if topo[k+direc[x]][j+direc[y]][i+direc[z]] == 1:
                                            if topo[k+direc[x]][j][i] == 0 and topo[k][j+direc[y]][i] == 0 and topo[k][j][i+direc[z]] == 0:
                                                if topo[k+direc[x]][j+direc[y]][i] == 0 and topo[k+direc[x]][j][i+direc[z]] == 0 and topo[k][j+direc[y]][i+direc[z]] == 0:
                                                    # topo[k][j][i] = 0
                                                    topcrs = 1
                                                    flag = 1
                                                    break
                                if topcrs == 1:
                                    break
                            if topcrs == 1:
                                break
                        if topcrs == 1:
                            return topcrs 
    return topcrs


def design_const(topologys):  # reduce topology from constraint (remove non-connected parts)
    topos = topologys.reshape((2, lz, ly, lx))  # correction: topo >> topos
    direc = [1, -1]  # correction: add direc
    for a in range(2):
        flag = 1
        while flag:
            # a=shuffle range(a)
            flag = 0
            for i in range(lx):
                for j in range(ly):
                    for k in range(lz):
                        topcrs = 0
                        if topos[a][k][j][i] == 1:
                            for x in range(2):
                                for y in range(2):
                                    if k + direc[x] in range(lz) and j + direc[y] in range(ly):  # correction: a,b,c >> lx, ly, lz
                                        if topos[a][k + direc[x]][j + direc[y]][i] == 1:
                                            if topos[a][k + direc[x]][j][i] == 0 and topos[a][k][j + direc[y]][i] == 0:
                                                topos[a][k][j][i] = 0
                                                topcrs = 1
                                                flag = 1
                                                break
                                    if k + direc[x] in range(lz) and i + direc[y] in range(lx):
                                        if topos[a][k + direc[x]][j][i + direc[y]] == 1:
                                            if topos[a][k + direc[x]][j][i] == 0 and topos[a][k][j][i + direc[y]] == 0:
                                                topos[a][k][j][i] = 0
                                                topcrs = 1
                                                flag = 1
                                                break
                                    if j + direc[x] in range(ly) and i + direc[y] in range(lx):
                                        if topos[a][k][j + direc[x]][i + direc[y]] == 1:
                                            if topos[a][k][j + direc[x]][i] == 0 and topos[a][k][j][i + direc[y]] == 0:
                                                topos[a][k][j][i] = 0
                                                topcrs = 1
                                                flag = 1
                                                break
                                    for z in range(2):
                                        if k + direc[x] in range(lz) and j + direc[y] in range(ly) and i + direc[z] in range(lx):
                                            if topos[a][k + direc[x]][j + direc[y]][i + direc[z]] == 1:
                                                if topos[a][k + direc[x]][j][i] == 0 and topos[a][k][j + direc[y]][i] == 0 and topos[a][k][j][i + direc[z]] == 0:
                                                    if topos[a][k + direc[x]][j + direc[y]][i] == 0 and topos[a][k + direc[x]][j][i + direc[z]] == 0 and topos[a][k][j + direc[y]][i + direc[z]] == 0:
                                                        topos[a][k][j][i] = 0
                                                        topcrs = 1
                                                        flag = 1
                                                        break
                                    if topcrs == 1:
                                        break
                                if topcrs == 1:
                                    break
    return topos


def inspect_clones(topologys, w):  # inspect topology clones
    if w == 1:
        flag1 = 0
    else:
        for j in range(1, w):
            topo = read_numpy_from_csv('topo_parent', j, from_type=int, to_type=np.float32)
            parents_topo = topo.reshape((end_pop, lz, ly, lx))
            for i in range(len(parents_topo)):
                if np.array_equal(parents_topo[i], topologys[0]) == True or np.array_equal(parents_topo[i],topologys[1]) == True:
                    flag1 = 1
                    break
                else:
                    flag1 = 0
                    break
            if flag1 == 1:
                break
    return flag1


def bfs_one(x, y, z, topo):
    direction = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    visited = np.zeros((lz, ly, lx))
    queue = deque()
    queue.append([x, y, z])
    visited[z][y][x] = 1
    exit_x0 = False
    exit_x = False
    exit_y0 = False
    exit_y = False
    exit_z0 = False
    exit_z = False

    while queue:
        x, y, z = queue.popleft()
        for i in range(6):
            nx = x + direction[i][0]
            ny = y + direction[i][1]
            nz = z + direction[i][2]
            if nx in range(lx) and ny in range(ly) and nz in range(lz):
                if topo[nz][ny][nx] == 1 and visited[nz][ny][nx] == 0:
                    visited[nz][ny][nx] = 1
                    queue.append([nx, ny, nz])
                    # endpoint = 0

                    if nz == lz - 1:
                        exit_z = True
                    if nx == lx - 1:
                        exit_x = True
                    if ny == ly - 1:
                        exit_y = True
                    if nz == 0:
                        exit_z0 = True
                    if ny == 0:
                        exit_y0 = True
                    if nx == 0:
                        exit_x0 = True

    if exit_x == True and exit_y0 == True and exit_y == True and exit_z0 == True and exit_z == True and exit_x0 == True:
        for x in range(lx):
            for y in range(ly):
                for z in range(lz):
                    if topo[z][y][x] == 1 and visited[z][y][x] == 0:
                        topo[z][y][x] = 0

        global topoend
        topoend = 1
        print('good')
        return topo

    else:
        print('Trapped!')


def bfs(x, y, z, offsprings_mm):  # check and find the topology with some geometirc constraints;
    offspringss = offsprings_mm.reshape((2, lz, ly, lx))
    end = [0, 0]
    for j in range(2):
        direction = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        visited = np.zeros((lz, ly, lx))
        queue = deque()
        queue.append([x, y, z])
        visited[z][y][x] = 1
        exit_x0 = False
        exit_x = False
        exit_y0 = False
        exit_y = False
        exit_z0 = False
        exit_z = False

        while queue:
            x, y, z = queue.popleft()
            for i in range(6):
                nx = x + direction[i][0]
                ny = y + direction[i][1]
                nz = z + direction[i][2]

                if nx in range(lx) and ny in range(ly) and nz in range(lz):
                    if offspringss[j][nz][ny][nx] == 1 and visited[nz][ny][nx] == 0:
                        visited[nz][ny][nx] = 1
                        queue.append([nx, ny, nz])
                        # endpoint = 0

                        if nz == lz - 1:
                            exit_z = True
                        if nx == lx - 1:
                            exit_x = True
                        if ny == ly - 1:
                            exit_y = True
                        if nz == 0:
                            exit_z0 = True
                        if ny == 0:
                            exit_y0 = True
                        if nx == 0:
                            exit_x0 = True

        if exit_x == True and exit_y0 == True and exit_y == True and exit_z0 == True and exit_z == True and exit_x0 == True:
            end[j] = 1
            for x in range(lx):
                for y in range(ly):
                    for z in range(lz):
                        if offspringss[j][z][y][x] == 1 and visited[z][y][x] == 0:
                            offspringss[j][z][y][x] = 0

    if end[0] == 1 and end[1] == 1:
        global topoend
        topoend = 1
        print('good')
        return offspringss

    else:
        print('trapped')


def cutting_function(topologys):
    flag = 0
    cuttings = []

    for i in range(topologys.shape[1]):
        cuttings.append(i)
    random.shuffle(cuttings)

    while flag == 0:  ## selecte cutting point
        candidate = []
        cutting = cuttings.pop()

        for i in range(topologys.shape[0]):  # correction: topo > topologys
            if topologys[i, cutting] == 1:  # correction: topo > topologys
                candidate.append(i)

            if len(candidate) > 1:
                flag = 1

        if len(cuttings) == 0:
            print('cant make connection')
            exit()

    return cutting, candidate


def candidates(cutting, candidate):  # correction: input: cutting >> cutting, candidate
    cuttingZ = cutting // (lx * ly)  # correction: variable name 'candidates' >> 'candidates_reslt'
    cuttingY = (cutting % (lx * ly)) // lx
    cuttingX = (cutting % (lx * ly)) % lx

    candidates_reslt = list(itertools.combinations(candidate, 2))  ## possible candidate pair of parents
    random.shuffle(candidates_reslt)

    return cuttingX, cuttingY, cuttingZ, candidates_reslt


# _____________________________________________________________________________

def topo_import(w):
    return read_numpy_from_csv('topo_offspring', w, from_type=int, to_type=np.float32)


def parent_import(w):
    ## import parents CSV data
    topo = read_numpy_from_csv('topo_parent', w, from_type=int, to_type=np.float32)
    reslt = read_numpy_from_csv('Output_parent', w, from_type=np.float32, to_type=np.float32)

    if restart_pop == 0:
        return topo, reslt

    else:  # restart
        offspring = read_numpy_from_csv('topo_offspring', w, from_type=int, to_type=np.float32)
        return topo, reslt, offspring


def offspring_import(w):
    if mode == 'Random':
        file_name = 'topo_parent'
        topo_1 = read_numpy_from_csv(file_name, w, from_type=int, to_type=np.float32)

    else:
        file_name = 'topo_offspring'
        topo_1 = read_numpy_from_csv(file_name, w, from_type=int, to_type=np.float32)

    file_name = 'Output_offspring'
    reslt_1 = read_numpy_from_csv(file_name, w, from_type=np.float32, to_type=np.float32)

    return topo_1, reslt_1


def evaluation(q, k):
    fitness_values = np.zeros((2 * q, 2))

    if evaluation_version == 'ver1':
        for i in range(q):
            dis11 = reslt[i][0]
            dis22 = reslt[i][1]
            RF22 = reslt[i][4]
            fit_val1 = (RF22 / MaxRF22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

        for i in range(q, 2 * q):
            dis11 = reslt_1[i - q][0]
            dis22 = reslt_1[i - q][1]
            RF22 = reslt_1[i - q][4]
            fit_val1 = (RF22 / MaxRF22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo_1[i - q]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver2':
        for i in range(q):
            RF22 = reslt[i][4]
            fit_val1 = np.sum(topo[i]) / (lx * ly * lz)
            fitness_values[i][0] = fit_val1
            fit_val2 = RF22 / MaxRF22
            fitness_values[i][1] = fit_val2

        for i in range(q, 2 * q):
            RF22 = reslt_1[i - q][4]
            fit_val1 = np.sum(topo_1[i - q]) / (lx * ly * lz)
            fitness_values[i][0] = fit_val1
            fit_val2 = RF22 / MaxRF22
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


def evaluation2(topo2, reslt2, k):
    fitness_values = np.zeros((topo2.shape[0], 2))

    if evaluation_version == 'ver1':
        for i in range(topo2.shape[0]):
            dis11 = reslt2[i][0]
            dis22 = reslt2[i][1]
            RF22 = reslt2[i][4]
            fit_val1 = (RF22 / MaxRF22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][0] = fit_val1
            fit_val2 = - (dis11 / dis22) + k * (np.sum(topo2[i]) / (lx * ly * lz))
            fitness_values[i][1] = fit_val2

    if evaluation_version == 'ver2':
        for i in range(topo2.shape[0]):
            RF22 = reslt2[i][4]
            fit_val1 = np.sum(topo2[i]) / (lx * ly * lz)
            fitness_values[i][0] = fit_val1
            fit_val2 = RF22 / MaxRF22
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
    selected_pop_index = np.zeros((number_solutions_needed))
    selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))

    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            # solution 1 is better than solution 2
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, (solution_1), axis=0)  # remove the selected solution
            fitness_values = np.delete(fitness_values, (solution_1),
                                       axis=0)  # remove the fitness of the selected solution
            crowding_distance = np.delete(crowding_distance, (solution_1),
                                          axis=0)  # remove the related crowding distance

        else:
            # solution 2 is better than solution 1
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, (solution_2), axis=0)
            fitness_values = np.delete(fitness_values, (solution_2), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_2), axis=0)

    selected_pop_index = np.asarray(selected_pop_index, dtype=int)  # Convert the data to integer
    return selected_pop_index


def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)  # initially assume all solutions are in pareto front by using "1"

    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0  # i is not in pareto front because j dominates i
                break  # no more comparision is needed to find out which one is dominated

    return pop_index[pareto_front]


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
            array_to_csv(f'Output_parent_{w + 1}.csv', reslt[i], dtype=np.float32, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topo[i], dtype=int, mode='a')

        if i >= end_pop:
            array_to_csv(f'Output_parent_{w + 1}.csv', reslt_1[i - end_pop], dtype=np.float32, mode='a')
            array_to_csv(f'topo_parent_{w + 1}.csv', topo[i - end_pop], dtype=int, mode='a')


def generate_offspring(w):
    global offspring
    while len(offspring) < end_pop:  # end_pop = populations
        candiat = 0
        while candiat == 0:  ## finding pair of offspring chromosomes
            cutting, candidate = cutting_function(topo)
            cuttingX, cuttingY, cuttingZ, candidates_reslt = candidates(cutting, candidate)
            print('candidate pair:', len(candidates_reslt))
            numberings = 0
            for j in range(len(candidates_reslt)):
                sames = 0
                global topoend  # correction: add 'global topoend'
                topoend = 0
                c1 = candidates_reslt[j][0]
                c2 = candidates_reslt[j][1]
                offsprings = crossover(topo[c1], topo[c2], cutting)  ## crossover stage
                offsprings_m = mutation(offsprings, mutation_rate)  ## mutation stage
                offsprings_mm = design_const(offsprings_m)  # offsprings > offsprings_m
                flag2 = inspect_clones(offsprings_mm, w)
                if flag2 == 1:
                    print('clone structure')
                    continue
                offsprings_mmm = offsprings_mm.reshape((2, lz, ly, lx))
                y0_num = np.where(offsprings_mmm[0][:, 0, :] == 1)
                if len(y0_num[0]) == 0:
                    continue
                offsprings_mmm0 = bfs_total(offsprings_mmm[0])
                numberings += 1
                print('bfs:', numberings)
                if topoend == 0:
                    continue
                else:
                    if len(offspring) == 0:
                        offspring = np.append(offspring, [offsprings_mmm0], axis=0)
                        candiat = 1
                        off_flt = offsprings_mmm0.flatten()
                        array_to_csv(f'topo_offspring_{w}.csv', off_flt, dtype=int, mode='a', save_as_int=True)
                        print('different')
                        break
                    else:
                        for i in range(len(offspring)):
                            if np.array_equal(offspring[i], offsprings_mmm0) == True:
                                print('same')
                                sames = 1
                                break
                        if sames == 0:  # save offspring topology
                            offspring = np.append(offspring, [offsprings_mmm0], axis=0)
                            candiat = 1
                            print('different')
                            off_flt = offsprings_mmm0.flatten()
                            array_to_csv(f'topo_offspring_{w}.csv', off_flt, dtype=int, mode='a', save_as_int=True)
                            break
                topoend = 0
                y0_num = np.where(offsprings_mmm[1][:, 0, :] == 1)
                if len(y0_num[0]) == 0:
                    continue
                offsprings_mmm1 = bfs_total(offsprings_mmm[1])
                numberings += 1
                print('bfs:', numberings)
                if topoend == 0:
                    continue
                else:
                    if len(offspring) == 0:
                        offspring = np.append(offspring, offsprings_mmm1, axis=0)
                        candiat = 1
                        off_flt = offsprings_mmm1.flatten()
                        array_to_csv(f'topo_offspring_{w}.csv', off_flt, dtype=int, mode='a', save_as_int=True)
                        print('different')
                        break
                    else:
                        for i in range(len(offspring)):
                            if np.array_equal(offspring[i], offsprings_mmm1) == True:
                                print('same')
                                sames = 1
                                break
                        if sames == 0:  # save offspring topology
                            offspring = np.append(offspring, offsprings_mmm1, axis=0)
                            candiat = 1
                            print('different')
                            off_flt = offsprings_mmm1.flatten()
                            array_to_csv(f'topo_offspring_{w}.csv', off_flt, dtype=int, mode='a', save_as_int=True)
                            break
                # offspringss = np.stack([offsprings_mmm0,offsprings_mmm1],axis=0)
            # if candiat == 1:
            #     break
        print('offsprings', len(offspring))
    return offspring


def generate_topo(q):
    if q == 1:
        topologys = np.random.randint(2, size=(lz, ly, lx))  ##topo [c][b][a]
        # topo2=topologys.flatten()
        # e2 = open('topo_parent_1_temp.csv', 'a')
        # f1 = open('topology.csv', 'a', encoding='utf-8', newline='')
        # wr = csv.writer(e2)
        # wr.writerow(topo2)
        # e2.close()

    else:
        same = 1
        topo1 = read_numpy_from_csv('topo_parent', 1, from_type=int, to_type=np.float32)
        endpop = len(topo1)
        parents_topo = topo1.reshape((endpop, lz, ly, lx))

        while same:
            same = 0
            topologys = np.random.randint(2, size=(lz, ly, lx))
            for i in range(endpop):
                if np.array_equal(parents_topo[i], topologys) == True:
                    same = 1
                    break

    return topologys


def generate_random(q):
    global topoend
    topoend = 1

    while topoend:
        topo = generate_topo(q)
        topo_1 = design_const_one(topo)
        flag = 1
        while flag:
            for x in range(lx):
                for y in range(ly):
                    for z in range(lz):
                        if topo_1[z][y][x] == 1:
                            topo_2 = bfs_one(x, y, z, topo_1)
                            flag = 0
                            break

                    if flag == 0:
                        break

                if flag == 0:
                    break

    array_to_csv('topo_parent_1.csv', topo_2.flatten(), dtype=int, mode='a')
    return topo_2


def generate_topo_bfs_one():
    direction = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    visited = np.zeros((lz, ly, lx))
    queue = deque()
    initiate_y = np.random.randint(ly);
    initiate_z = np.random.randint(lz)
    queue.append([0, initiate_y, initiate_z])
    visited[initiate_z][initiate_y][0] = 1
    topo_number = 1

    # exit_x0 = False
    exit_x = False
    exit_y0 = False
    exit_y = False
    exit_z0 = False
    exit_z = False

    visit_dir = []

    while queue:
        end_count = 0
        x, y, z = queue.popleft()
        nextdir = np.random.randint(6)

        if nextdir not in visit_dir:
            visit_dir.append(nextdir)

        else:
            queue.append([x, y, z])
            continue

        nx = x + direction[nextdir][0]
        ny = y + direction[nextdir][1]
        nz = z + direction[nextdir][2]

        if nx in range(lx) and ny in range(ly) and nz in range(lz):
            flag = design_const_voxel(nx, ny, nz, visited)

            if flag == 0:
                if visited[nz][ny][nx] == 0:
                    visited[nz][ny][nx] = 1
                    topo_number += 1
                    queue.append([nx, ny, nz])
                    visit_dir = []

                else:
                    queue.append([nx, ny, nz])
                    visit_dir = []

                if nx == lx - 1:
                    exit_x = True
                if ny == lx - 1:
                    exit_y = True
                if nz == lz - 1:
                    exit_z = True
                if ny == 0:
                    exit_y0 = True
                if nz == 0:
                    exit_z0 = True
                if exit_x == True and exit_y == True and exit_y0 == True and exit_z0 == True and exit_z == True:
                    possible = np.random.randint(3)
                    if possible == 2:
                        break
            else:
                queue.append([x, y, z])
        else:
            queue.append([x, y, z])

    return visited


def generate_topo_bfs():
    direction = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

    visited = np.zeros((lz, ly, lx))

    queue = deque()

    initiate_y = np.random.randint(ly);
    initiate_z = np.random.randint(lz)

    queue.append([0, initiate_y, initiate_z])

    visited[initiate_z][initiate_y][0] = 1

    topo_number = 1

    # exit_x0 = False

    exit_x = False

    exit_y0 = False

    exit_y = False

    exit_z0 = False

    exit_z = False

    visit_dir = []

    while queue:

        end_count = 0

        x, y, z = queue.popleft()

        nextdir = np.random.randint(6)

        if nextdir not in visit_dir:

            visit_dir.append(nextdir)

        else:

            queue.append([x, y, z])

            continue

        nx = x + direction[nextdir][0]

        ny = y + direction[nextdir][1]

        nz = z + direction[nextdir][2]

        if nx in range(lx) and ny in range(ly) and nz in range(lz):

            flag = design_const_voxel(nx, ny, nz, visited)

            if flag == 0:

                if visited[nz][ny][nx] == 0:

                    visited[nz][ny][nx] = 1

                    topo_number += 1

                    queue.append([nx, ny, nz])

                    visit_dir = []

                else:

                    queue.append([nx, ny, nz])

                    visit_dir = []

                if nx == lx - 1:
                    exit_x = True

                if ny == lx - 1:
                    exit_y = True

                if nz == lz - 1:
                    exit_z = True

                if ny == 0:
                    exit_y0 = True

                if nz == 0:
                    exit_z0 = True

                if exit_x == True and exit_y == True and exit_y0 == True and exit_z0 == True and exit_z == True:

                    possible = np.random.randint(3)

                    if possible == 2:
                        break

            else:

                queue.append([x, y, z])

        else:

            queue.append([x, y, z])

    return visited


def design_const_voxel(nx, ny, nz, topo):
    flag = 0

    direc = [1, -1]

    for x in range(2):

        for y in range(2):

            if nz + direc[x] in range(lz) and ny + direc[y] in range(ly):

                if topo[nz + direc[x]][ny + direc[y]][nx] == 1:

                    if topo[nz + direc[x]][ny][nx] == 0 and topo[nz][ny + direc[y]][nx] == 0:
                        flag = 1

                        break

            if nz + direc[x] in range(lz) and nx + direc[y] in range(lx):

                if topo[nz + direc[x]][ny][nx + direc[y]] == 1:

                    if topo[nz + direc[x]][ny][nx] == 0 and topo[nz][ny][nx + direc[y]] == 0:
                        flag = 1

                        break

            if ny + direc[x] in range(ly) and nx + direc[y] in range(lx):

                if topo[nz][ny + direc[x]][nx + direc[y]] == 1:

                    if topo[nz][ny + direc[x]][nx] == 0 and topo[nz][ny][nx + direc[y]] == 0:
                        flag = 1

                        break

            for z in range(2):
                if nz + direc[x] in range(lz) and ny + direc[y] in range(ly) and nx + direc[z] in range(lx):
                    if topo[nz + direc[x]][ny + direc[y]][nx + direc[z]] == 1:
                        if topo[nz + direc[x]][ny][nx] == 0 and topo[nz][ny + direc[y]][nx] == 0 and topo[nz][ny][
                            nx + direc[z]] == 0:
                            if topo[nz + direc[x]][ny + direc[y]][nx] == 0 and topo[nz + direc[x]][ny][
                                nx + direc[z]] == 0 and topo[nz][ny + direc[y]][nx + direc[z]] == 0:
                                flag = 1
                                break

            if flag == 1:
                break

        if flag == 1:
            break

    return flag


def design_const_add(topologys):
    topo = topologys.copy()

    flag = 1

    direc = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

    while flag:

        # a=shuffle range(a)

        flag = 0

        for i in range(lx):

            for j in range(ly):

                for k in range(lz):

                    topcrs = 0

                    if topo[k][j][i] == 1:

                        for m in range(4):

                            if k + direc[m][0] in range(lz) and j + direc[m][1] in range(ly):

                                if topo[k + direc[m][0]][j + direc[m][1]][i] == 1:

                                    if topo[k + direc[m][0]][j][i] == 0 and topo[k][j + direc[m][1]][i] == 0:

                                        randbit = random.getrandbits(1)

                                        if randbit == 1:

                                            topo[k + direc[m][0]][j][i] = 1

                                        else:

                                            topo[k][j + direc[m][1]][i] = 1

                                        topcrs = 1

                                        flag = 1

                                        break

                            if k + direc[m][0] in range(lz) and i + direc[m][1] in range(lx):

                                if topo[k + direc[m][0]][j][i + direc[m][1]] == 1:

                                    if topo[k + direc[m][0]][j][i] == 0 and topo[k][j][i + direc[m][1]] == 0:

                                        randbit = random.getrandbits(1)

                                        if randbit == 1:

                                            topo[k + direc[m][0]][j][i] = 1

                                        else:

                                            topo[k][j][i + direc[m][1]] = 1

                                        topcrs = 1

                                        flag = 1

                                        break

                            if j + direc[m][0] in range(ly) and i + direc[m][1] in range(lx):

                                if topo[k][j + direc[m][0]][i + direc[m][1]] == 1:

                                    if topo[k][j + direc[m][0]][i] == 0 and topo[k][j][i + direc[m][1]] == 0:

                                        randbit = random.getrandbits(1)

                                        if randbit == 1:

                                            topo[k][j + direc[m][0]][i] = 1

                                        else:

                                            topo[k][j][i + direc[m][1]] = 1

                                        topcrs = 1

                                        flag = 1

                                        break

                            for z in [1, -1]:

                                if k + direc[m][0] in range(lz) and j + direc[m][1] in range(ly) and i + z in range(lx):

                                    if topo[k + direc[m][0]][j + direc[m][1]][i + z] == 1:

                                        if topo[k + direc[m][0]][j][i] == 0 and topo[k][j + direc[m][1]][i] == 0 and \
                                                topo[k][j][i + z] == 0:

                                            if topo[k + direc[m][0]][j + direc[m][1]][i] == 0 and \
                                                    topo[k + direc[m][0]][j][i + z] == 0 and topo[k][j + direc[m][1]][
                                                i + z] == 0:

                                                rand1 = random.randint(1, 3)

                                                rand2 = random.randint(1, 2)

                                                if rand1 == 1:

                                                    topo[k + direc[m][0]][j][i] = 1

                                                    if rand2 == 1:

                                                        topo[k + direc[m][0]][j + direc[m][1]][i] = 1

                                                    else:

                                                        topo[k + direc[m][0]][j][i + z] = 1

                                                if rand1 == 2:

                                                    topo[k][j + direc[m][1]][i] = 1

                                                    if rand2 == 1:

                                                        topo[k + direc[m][0]][j + direc[m][1]][i] = 1

                                                    else:

                                                        topo[k][j + direc[m][1]][i + z] = 1

                                                if rand1 == 3:

                                                    topo[k][j][i + z] = 1

                                                    if rand2 == 1:

                                                        topo[k + direc[m][0]][j][i + z] = 1

                                                    else:

                                                        topo[k][j + direc[m][1]][i + z] = 1

                                                topcrs = 1

                                                flag = 1

    return topo


def generate_random_bfs(q):
    if q == 1:
        topologys = generate_topo_bfs()

    else:
        same = 1
        topo1 = read_numpy_from_csv('topo_parent', 1, from_type=int, to_type=np.float32)
        endpop = len(topo1)
        parents_topo = topo1.reshape((endpop, lz, ly, lx))
        while same:
            same = 0
            topologys = generate_topo_bfs()
            for i in range(endpop):
                if np.array_equal(parents_topo[i], topologys) == True:
                    same = 1
                    break

    topo2 = topologys.flatten()
    array_to_csv('topo_parent_1.csv', topo2, dtype=int, mode='a')
    return topologys


def visualize(w):
    topo_p, reslt_p = parent_import(w + 1)
    fitness_values = evaluation2(topo_p, reslt_p)
    index = np.arange(topo_p.shape[0]).astype(int)
    pareto_front_index = pareto_front_finding(fitness_values, index)
    index = reslt_p[pareto_front_index, :]  # correction: index >> reslt_p
    fitness_values_pareto = fitness_values[pareto_front_index]
    fv1_sort, fv2_sort, sort = fitval_sort(fitness_values_pareto)
    normalized_hypervolume, hypervolume = hypervolume_calculation(fv1_sort, fv2_sort)

    print('\n')
    print("iteration: %d" % (w))
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
        gen_num, pop_num = find_model_output(w + 1, sort[q])
        print("%dth generation %dth pop   [%E   %E]   Job-%d-%d" % (
        w + 1, q + 1, fitness_values[q, 0], fitness_values[q, 1], gen_num, pop_num))
    # print(fitness_values)
    ##pareto_front
    fit_max, fit_min = max_fitval(end_gen)
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


##Overall process for design
# with open('topo_parent_1_temp.csv', 'r') as in_file:  ##delete empty row
#        with open('topo_parent_1.csv', 'wb') as out_file:
#            writer = csv.writer(out_file)
#            for row in csv.reader(in_file):
#               if row:
#                    writer.writerow(row)


# with open('Output_random_1_temp.csv' , 'r') as in_file:  ##delete empty row
#        with open('Output_parent_1.csv', 'wb') as out_file:
#            writer = csv.writer(out_file)
#            for row in csv.reader(in_file):
#                if row:
#                    writer.writerow(row)

if __name__ == '__main__':
    # Open ABAQUS until the main function of this script ends
    th = run_abaqus_script_without_gui(set_path=setPath, abaqus_script_name=abaqus_script_name, params=PARAMS,
                                       abaqus_execution_mode=abaqus_execution_mode) # Define an abaqus thread
    th.daemon = False
    # if daemon == True, ABAQUS exits when this python process is terminated
    # if daemon == False, ABAQUS doesn't exit when this python process is terminated, and this is useful for debugging
    th.start() # start abaqus thread
    # For-loop of abaqus jobs
    if mode == 'GA':
        if restart_pop == 0:
            for w in range(ini_gen, end_gen + 1):
                topo, reslt = parent_import(w)
                offspring = np.empty((0, lz, ly, lx), int)
                offspring = generate_offspring(w)

                # ********** start of an ABAQUS job **********
                args = {
                    'order': 1,
                    'w': w,
                    'offspring': offspring,
                }
                with open('./args', mode='wb') as f_args:
                    pickle.dump(args, f_args, protocol=2)
                wait_for_abaqus_job_done(check_exit_time=1)
                # ********** end of an ABAQUS job **********

                topo_1, reslt_1 = offspring_import(w)  # data import
                fitness_values = evaluation(end_pop, penalty_coefficient)  # calculate fitness values
                save_variable_for_debugging(debug_code=1, w=w, debug_variable=[topo_1, reslt_1, fitness_values])

                pop = np.append(topo, topo_1, axis=0)  # integrated pop(topology data)
                save_variable_for_debugging(debug_code=2, w=w, debug_variable=pop)

                pop, next_generations = selection(pop, fitness_values, end_pop)  # selection (index)
                save_variable_for_debugging(debug_code=3, w=w, debug_variable=[pop, next_generations])

                parent_export(w)

                print('iteration:', w)

        else:
            w = ini_gen
            topo, reslt, offspring = parent_import(w)
            offspring = offspring.reshape((end_pop, lx, ly, lz))
            topo_1, reslt_1 = offspring_import(w)  # data import
            fitness_values = evaluation(end_pop, penalty_coefficient)  # calculate fitness values
            pop = np.append(topo, topo_1, axis=0)  # integrated pop(topology data)
            pop, next_generations = selection(pop, fitness_values, end_pop)  # selection (index)
            parent_export(w)
            print('iteration:', w)
            restart_pop = 0
            for w in range(ini_gen + 1, end_gen + 1):
                topo, reslt = parent_import(w)
                offspring = np.empty((0, lz, ly, lx), int)
                offspring = generate_offspring(w)

                # ********** start of an ABAQUS job **********
                args = {
                    'order': 2,
                    'w': w,
                    'offspring': offspring,
                }
                with open('./args', mode='wb') as f_args:
                    pickle.dump(args, f_args, protocol=2)
                wait_for_abaqus_job_done(check_exit_time=1)
                # ********** end of an ABAQUS job **********

                topo_1, reslt_1 = offspring_import(w)  # data import
                fitness_values = evaluation(end_pop, penalty_coefficient)  # calculate fitness values
                save_variable_for_debugging(debug_code=4, w=w, debug_variable=[topo_1, reslt_1, fitness_values])

                pop = np.append(topo, topo_1, axis=0)  # integrated pop(topology data)
                save_variable_for_debugging(debug_code=5, w=w, debug_variable=pop)

                pop, next_generations = selection(pop, fitness_values, end_pop)  # selection (index)
                save_variable_for_debugging(debug_code=6, w=w, debug_variable=[pop, next_generations])

                parent_export(w)
                print('iteration:', w)

            # visualize(end_gen)

    with open('./args_end', mode='wb') as f_args_end:
        pickle.dump('end', f_args_end, protocol=2)
    th.join() # this python script exits only when abaqus process is closed manually.