import string, time
from random import random
import numpy as np
from sklearn.decomposition import PCA
from pprint import pprint as pp
import multiprocessing, sys, os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import rc,rcParams
import matplotlib
import time
start=time.time()

sys.setrecursionlimit(64 * 64 * 1000)

ptrain = [0.0 + x*0.025 for x in range(41)]
# ptrain = [0.3,0.8]
ptest = ptrain
plist = ptest
LENGTH_GET = [10,20,30,40]
# LENGTH_GET = [4,6]
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
   
# for p in np.linspace(0, 1, 41):
#     p = round(p, 3)
#     time1 = time.time()
#     pool = multiprocessing.Pool(1)
#     cell = grid(p)
#     time2 = time.time()
#     print(time2 - time1)
result = []
for size in range(len(LENGTH_GET)):
    N = LENGTH_GET[size]
    t = 500
    a_test = []
    for p in ptest:
        pool = multiprocessing.Pool(10)
        cell = grid(p)
        print(cell)
        print(type(cell))
        print(cell.shape)
        maxtreeresult = []
        for i in range(t):
            try:
                new_cell = pool.apply(maxtree, (cell[i],))
            except:
                new_cell = maxtree(cell[i])
            maxtreeresult .append(new_cell)
        print(maxtreeresult )
        print(type(maxtreeresult ))
        AO = np.array(maxtreeresult )
        print(type(AO))
        print(AO.shape)
        A = AO.reshape(t,-1)
        print(A)
        print(A.shape)
        # print(B.shape)
        # plt.imshow(np.array(B))
        # plt.show()
        AA = A.tolist()
        print(type(AA)) 
        a_test.append(AA)  
    print(a_test)

    XO = np.array(a_test)
    X = XO.reshape(len(ptest)*t,-1)
    print(X.shape)
    pca = PCA(n_components = 2, svd_solver='full')
    pca.fit(X)
    X_reduction = pca.transform(X)
    print(pca.explained_variance_ratio_) 
    print(X_reduction)
    print(X_reduction.shape)
    # XXXXXX= X_reduction/LENGTH
    print (pca.n_components_)
    XX = X_reduction[:,0]
    print(XX)
    print(XX.shape)
    XXX = np.array(XX).reshape(len(ptrain),t)
    print(XXX)
    print(XXX.shape)
    XXXX = XXX.mean(axis=1)# print(c.mean(axis=1))#行# print(c.mean(axis=0))#列
    print(XXXX)
    print(XXXX.shape)
    XXXXX = XXXX.tolist()
    # colors = ['navy', 'turquoise', 'darkorange']
    # To make plots pretty
    # print(plist[0:])
    print(XXXXX)

    result.append(XXXXX)
print(result)

final = np.array(result)
print(final.shape)
print(final[0,:])
print(final[3,:])

golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
# cm = plt.cm.get_cmap('rainbow')
# plt.rc('font',**{'size':16})
# fig, ax = plt.subplots(1,figsize=golden_size(8))
colors = ['red', 'orange', 'green',  'blue']
GET = ['L = 10', 'L = 20', 'L = 30', 'L = 40']
# GET = ['L = 50', 'L = 100', 'L = 150', 'L = 200']
plt.figure(figsize=golden_size(12))
for i in range(len(LENGTH_GET)):
    G  = GET[i]
    plt.plot(plist[0:], final[i,:], marker='o', linewidth=3, label = G)
    # plt.scatter(plist[0:], final[size,:], s=10, marker='o', linewidth=3)


plt.xlabel('${p}$',fontsize=20)
# plt.ylabel('${<p_1>/L}$')
plt.ylabel('${<p_1>}$',fontsize=20)
plt.tick_params(axis='both',which='both',direction='in')
# # plt.colorbar(sc, label='$0.25\\times$Temperature')
# # plt.colorbar(sc)
plt.legend()
plt.savefig('pca_p1_pc_maxtree.pdf')

end=time.time()
print('Running time: %s Seconds'%(end-start))

plt.show()
