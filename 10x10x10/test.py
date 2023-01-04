import numpy as np
import numba
from numba import njit, jit
import pickle


@njit
def f():
    a = np.empty(shape=(0, 3), dtype=numba.int64)  # deque()
    a = np.array([[1, 2, 3]], dtype=numba.int64)  # deque() and append()
    a = np.vstack((a, np.array([[0, 5, 6]])))  # append()
    pop = a[0]; a = a[1:]  # popleft

    print(a)


@njit
def bfs_alldirec(x, y, z, lx, ly, lz, topology, topoend):
    topo = topology.copy().astype(numba.int64)
    direction = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    visited = np.zeros((lz, ly, lx), dtype=numba.int64)
    queue = np.array([[x, y, z]], dtype=numba.int64)  # deque()
    queue = np.vstack((queue, np.array([[x, y, z]])))  # append()
    visited[z][y][x] = 1
    exit_x0 = False
    exit_x = False
    exit_y0 = False
    exit_y = False
    exit_z0 = False
    exit_z = False

    while len(queue):
        x, y, z = queue[0]; queue = queue[1:]  # popleft
        for i in range(6):
            nx = x + direction[i][0]
            ny = y + direction[i][1]
            nz = z + direction[i][2]
            if nx in range(lx) and ny in range(ly) and nz in range(lz):
                if topo[nz][ny][nx] == 1 and visited[nz][ny][nx] == 0:
                    visited[nz][ny][nx] = 1
                    queue = np.vstack((queue, np.array([[nx, ny, nz]])))  # append()
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

    if exit_x and exit_x0 and exit_y and exit_y0 and exit_z and exit_z0:
        for x in range(lx):
            for y in range(ly):
                for z in range(lz):
                    if topo[z][y][x] == 1 and visited[z][y][x] == 0:
                        topo[z][y][x] = 0
        topoend = 1
        print('good')
        # print('topo_refined',len(np.where(topo<0)[0]))
        return topo, topoend
    else:
        print('Trapped!')
    return None, topoend


@njit
def bfs_ydirec(x, y, z, lx, ly, lz, topology):
    print('ydirec')
    topo = topology.copy()
    direction = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
    visited = np.zeros((lz, ly, lx),dtype=numba.int64)
    queue = np.array([[x, y, z]], dtype=numba.int64)  # deque() and append()
    queue_back = np.empty(shape=(0, 3), dtype=numba.int64)  # deque()
    visited[z][y][x] = 1
    while len(queue):
        x, y, z = queue[0]; queue = queue[1:]  # popleft
        path_count = 0
        for i in range(5):
            nx = x + direction[i, 0]
            ny = y + direction[i, 1]
            nz = z + direction[i, 2]
            if nx in range(lx) and ny in range(ly) and nz in range(lz):
                if topo[nz][ny][nx] == 1 and visited[nz][ny][nx] == 0:
                    visited[nz][ny][nx] = 1
                    queue = np.vstack((queue, np.array([[nx, ny, nz]])))  # append()
                    path_count += 1

        if path_count == 0:
            if y == ly - 2:
                visited[z][y + 1][x] = 1  # adding voxel for connection
            if y < ly - 2:
                queue_back = np.vstack((queue_back, np.array([[x, y, z]])))  # append()

    # print('y_direc_first_queue_end')
    while len(queue_back):
        x, y, z = queue_back[0]; queue_back = queue_back[1:]  # popleft
        visited = bfs_back(x, y, z, lx, ly, lz, visited)
    # print('y_direc_queue_back_end')
    ly_num = np.where(visited[:, ly - 1, :] == 1)
    if len(ly_num[0]) == 0:
        visited = np.zeros((lz, ly, lx), dtype=numba.int64)
    return visited


@njit
def bfs_back(x, y, z, lx, ly, lz, topology):
    topo = topology.copy()
    direction = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]
    visited = np.zeros((lz, ly, lx), dtype=numba.int64)
    queue = np.array([[x, y, z]], dtype=numba.int64)  # deque() and append()
    visited[z][y][x] = 1
    queue2 = np.empty(shape=(0, 3), dtype=numba.int64)  # deque()
    y1 = y
    while len(queue):
        x, y, z = queue[0]; queue = queue[1:]  # popleft
        for i in range(5):
            nx = x + direction[i][0]
            ny = y + direction[i][1]
            nz = z + direction[i][2]
            if nx in range(lx) and ny in range(ly) and nz in range(lz):
                if topo[nz][ny][nx] == 1 and visited[nz][ny][nx] == 0:
                    visited[nz][ny][nx] = 1
                    queue = np.vstack((queue, np.array([[nx, ny, nz]])))  # append()
        if y == y1 and topo[z][y1 - 1][x] == 1:
            queue2 = np.vstack((queue2, np.array([[x, y1 - 1, z]])))  # append()
    # print('bfs_back_queue_end')

    ly_num = np.where(visited[:, ly - 1, :] == 1)
    if len(ly_num[0]) != 0:
        return topo
    if len(ly_num[0]) == 0:
        topo - visited
        # print('bfsback_one_negative',len(np.where(topo<0)[0]))
    while len(queue2):
        queue = np.empty(shape=(0, 3), dtype=numba.int64)  # deque()
        visited = np.zeros((lz, ly, lx), dtype=numba.int64)
        x, y, z = queue2[0]; queue2 = queue2[1:]  # popleft
        y1 = y
        x1 = x
        z1 = z

        queue = np.vstack((queue, np.array([[x, y, z]])))  # append()
        if topo[z][y][x] == 1:
            while len(queue):
                x, y, z = queue[0]; queue = queue2[1:]  # popleft
                for i in range(5):
                    nx = x + direction[i][0]
                    ny = y + direction[i][1]
                    nz = z + direction[i][2]
                    if nx in range(lx) and ny in range(ly) and nz in range(lz):
                        if topo[nz][ny][nx] + visited[nz][ny][nx] == 0:
                            visited[nz][ny][nx] = 1
                            queue = np.vstack((queue, np.array([[nx, ny, nz]])))  # append()
            # print('bfs_back_queue2_queue_end')

        ly_num = np.where(visited[:, ly - 1, :] == 1)

        if len(ly_num[0]) != 0:
            break

        else:
            if topo[z1][y1 - 1][x1] == 1:
                queue2 = np.vstack((queue2, np.array([[x1, y1 - 1, z1]])))  # append()
            topo - visited
            # print('bfsback_two_negative',len(np.where(topo<0)[0]))
    # print('bfs_back_queue2_end')
    return topo


@njit
def design_const_one(topo, lx, ly, lz):
    flag = 1
    direc = [1, -1]
    while flag:
        # a=shuffle range(a)
        flag = 0
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    topcrs = 0
                    if topo[k][j][i] == 1:
                        for x in range(2):
                            for y in range(2):
                                if k + direc[x] in range(lz) and j + direc[y] in range(ly):
                                    if topo[k + direc[x]][j + direc[y]][i] == 1:
                                        if topo[k + direc[x]][j][i] == 0 and topo[k][j + direc[y]][i] == 0:
                                            # topo[k][j][i] = 0
                                            topcrs = 1
                                            flag = 1
                                            break

                                if k + direc[x] in range(lz) and i + direc[y] in range(lx):
                                    if topo[k + direc[x]][j][i + direc[y]] == 1:
                                        if topo[k + direc[x]][j][i] == 0 and topo[k][j][i + direc[y]] == 0:
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
                                    if k + direc[x] in range(lz) and j + direc[y] in range(ly) and i + direc[
                                        z] in range(lx):
                                        if topo[k + direc[x]][j + direc[y]][i + direc[z]] == 1:
                                            if topo[k + direc[x]][j][i] == 0 and topo[k][j + direc[y]][i] == 0 and \
                                                    topo[k][j][i + direc[z]] == 0:
                                                if topo[k + direc[x]][j + direc[y]][i] == 0 and topo[k + direc[x]][j][
                                                    i + direc[z]] == 0 and topo[k][j + direc[y]][i + direc[z]] == 0:
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


@njit
def bfs_total(topo, lx, ly, lz, topoend):
    global_visited = np.zeros((lz, ly, lx), dtype=numba.int64)
    topo_refined = np.zeros((lz, ly, lx), dtype=numba.int64)
    b = np.where(topo[:, 0, :] == 1)
    # print('first_topo_y0_number : ',len(b[0]))\
    for i in range(len(b[0])):
        if topoend == 1:
            topo_refined, topoend = bfs_alldirec(b[1][i], 0, b[0][i], lx, ly, lz, topo, topoend=topoend)
            break
    if topoend != 1:
        return None, topoend

    bb = np.where(topo_refined[:, 0, :] == 1)
    # print('first_ydirec_loop :', i)
    for i in range(len(bb[0])):
        if topo[b[0][i]][0][b[1][i]] == 1 and global_visited[b[0][i]][0][b[1][i]] == 0:
            global_visited += bfs_ydirec(b[1][i], 0, b[0][i], lx, ly, lz, topo_refined)
    global_visited = np.where(global_visited > 0, 1, global_visited)
    # print('global_visited_negative',len(np.where(global_visited<0)[0]))
    topo_crs = design_const_one(global_visited, lx, ly, lz)
    if topo_crs == 1:
        print('design is constrained')
        topoend = 0
        return None, topoend
    else:
        print('design is verified')
        return global_visited, topoend


if __name__ == '__main__':
    # with open('topo', mode='rb') as f:
    #     topo = pickle.load(f)
    # with open('topoend', mode='rb') as f:
    #     topoend = pickle.load(f)
    # # bfs_total(topo, 10, 10, 10, topoend=topoend)
    # # design_const_one(topo, 10, 10, 10)
    # bfs_total(topo, 10, 10, 10, topoend=topoend)
    f()