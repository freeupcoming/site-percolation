import string, time
from random import random
import numpy as np
from pprint import pprint as pp
import multiprocessing, sys, os
import matplotlib.pyplot as plt

sys.setrecursionlimit(64 * 64 * 1000)

N = 20
t = 10

NOT_VISITED = 1


def grid(p):
    cell = np.random.random([t, N, N])
    gird = cell < p
    return gird.astype(int)


def maxtree(grid):
    max_grid = np.zeros(np.shape(grid))
    for j in range(N):
        for i in range(N):
            new_grid = np.zeros(np.shape(grid))
            if grid[j][i] == NOT_VISITED:
                dfs(grid, new_grid, j, i)
            if np.sum(max_grid) < np.sum(new_grid):
                max_grid = new_grid
    return max_grid.astype(int)


def dfs(grid, new_grid, i, j):
    grid[i][j] = 0
    new_grid[i][j] = 1
    if i > 0 and grid[i - 1][j] == NOT_VISITED:
        dfs(grid, new_grid, i - 1, j)
    if i < N - 1 and grid[i + 1][j] == NOT_VISITED:
        dfs(grid, new_grid, i + 1, j)
    if j < N - 1 and grid[i][j + 1] == NOT_VISITED:
        dfs(grid, new_grid, i, j + 1)
    if j > 0 and grid[i][j - 1] == NOT_VISITED:
        dfs(grid, new_grid, i, j - 1)


def shuffle_part(vector, p_choose):
    y2 = []
    for i in vector:
        y2.extend(i)
    point_num = int(len(y2) * p_choose)
    nums = np.random.random(vector.shape[1] * vector.shape[0])
    index = []
    point = []
    # print('选择打乱顺序的节点数目：', point_num)
    for i in np.argsort(nums)[0:point_num]:
        index0 = [i // vector.shape[1], i % vector.shape[1]]
        index.append(index0)
        yy0 = vector[index0[0], index0[1]]
        point.append(yy0)
    # print('选择的节点坐标:', len(index))
    # print(point)
    np.random.shuffle(point)
    # print(point)
    for i in range(0, len(point)):
        vector[index[i][0], index[i][1]] = point[i]
    return vector


if __name__ == '__main__':
    for p in np.linspace(0, 1, 101):
        p = round(p, 3)
        time1 = time.time()
        pool = multiprocessing.Pool(20)

        cell = grid(p)
        if not os.path.exists('./data' + '/' + str(N)):
            os.makedirs('./data' + '/' + str(N))
        else:
            if not os.path.exists('./data' + '/' + str(N) + '/' + 'raw'):
                os.makedirs('./data' + '/' + str(N) + '/' + 'raw')
        # print(type(cell))
        np.save('./data' + '/' + str(N) + '/' + 'raw' + '/' + str(p) + '.npy', np.array(cell))

        result = []
        for i in range(t):
            try:
                new_cell = pool.apply(maxtree, (cell[i]),)
            except:
                new_cell = maxtree(cell[i])
            result.append(new_cell)
            if not os.path.exists('./data' + '/' + str(N)):
                os.makedirs('./data'  + '/'+ str(N))
            else:
                if not os.path.exists('./data' + '/' + str(N) + '/' + 'maxtree'):
                    os.makedirs('./data' + '/' + str(N) + '/' + 'maxtree')
        np.save('./data' + '/' + str(N) + '/' + 'maxtree' + '/' + str(p) + '.npy', np.array(result))

        result_shuffle = []
        for i in range(t):
            cell_shuffle = shuffle_part(result[i], p_choose)
            result_shuffle.append(cell_shuffle)
            if not os.path.exists('./data'  + '/'+ str(N)):
                os.makedirs('./data' + '/' + str(N))
            else:
                if not os.path.exists('./data' + '/' + str(N) + '/' + 'maxtree_shuffle'):
                    os.makedirs('./data' + '/' + str(N) + '/' + 'maxtree_shuffle')
        np.save('./data' + '/' + str(N) + '/' + 'maxtree_shuffle' + '/' + str(p) + '.npy', np.array(result_shuffle))

        result_shuffle_maxtree = []
        for i in range(t):
            cell_shuffle = result_shuffle[i]
            try:
                cell_shuffle_maxtree = pool.apply(maxtree, (cell_shuffle,))
            except:
                cell_shuffle_maxtree = maxtree(cell_shuffle)
            result_shuffle_maxtree.append(cell_shuffle_maxtree)
            if not os.path.exists('./data' + '/' + str(N)):
                os.makedirs('./data' + '/' + str(N))
            else:
                if not os.path.exists('./data' + '/' + str(N) + '/' + 'maxtree_shuffle_maxtree'):
                    os.makedirs('./data' + '/' + str(N) + '/' + 'maxtree_shuffle_maxtree')
        np.save('./data' + '/' + str(N) + '/' + 'maxtree_shuffle' + '/' + str(p)  + '.npy', np.array(result_shuffle_maxtree))

        time2 = time.time()
        print(time2 - time1)
