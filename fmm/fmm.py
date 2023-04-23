import numpy as np
import sys
sys.path.insert(0,'..')
from copy import deepcopy as dc
from collections import defaultdict
from itertools import product
from time import time
from scipy.linalg import cho_factor, cho_solve, cholesky, lu
from scipy.linalg.blas import dgemm

from problem_tools.geometry_tools import Data, Tree
from problem_tools.problem import Problem
# from functions import test_funcs
from numba import jit
from scipy.linalg.interpolative import interp_decomp
from scipy.sparse import csc_matrix as csc
from scipy.sparse import lil_matrix as lil
import scipy.sparse as sps
import warnings

import matplotlib.pyplot as plt
from memory_profiler import profile

class Fmm(object):
    def __init__(self,pr,proxy_p=1., proxy_r=1., symmetric_fun = 0, lr_type='id'):
        self.pr = pr
        self.proxy_p = proxy_p
        self.proxy_r = proxy_r
        self.tree = pr.tree
        self.n = pr.shape[0]
        tree = self.tree
        size = tree.level[-1]
        self.size = size
        self.basis = [None for i in range(size)]
        self.local_basis = [None for i in range(size)]
        self.T = [None for i in range(size)]
        self.index_lvl = [None for i in range(size)]
        self.elim_list = set()
        self.shape = self.pr.shape
        self.lr_type = lr_type
    def upd_index_lvl(self, ind):
        index_size = 0
        for ch in self.tree.child[ind]:
            index_size += self.basis[ch].shape[0]
        self.index_lvl[ind] = np.zeros(index_size, dtype=int)
        tmp = 0
        for ch in self.tree.child[ind]:
            self.index_lvl[ind][tmp:tmp+self.basis[ch].shape[0]] = self.basis[ch]
            tmp += self.basis[ch].shape[0]
    def dot(self, tmp_old):
        tmp = tmp_old.copy()
        pr = self.pr
        tree = pr.tree
        level_count = len(tree.level) - 2
        tl = self.tail_lvl
        ans = [np.zeros(pr.shape[0])] * (level_count - tl + 1)
        ans[0] = self.dot_cl(tmp)

        for i in range(level_count-1, tl-1, -1):
            tmp = self.dot_T(i, tmp, 'up')
            ans[level_count-i] = self.dot_il(i, tmp)
        for i in range(tl, level_count):
            ans[level_count-1-i] += self.dot_T(i, ans[level_count-i], 'down')
        return ans[0]
    def dot_cl(self, tmp):
        pr = self.pr
        tree = pr.tree
        level_count = len(tree.level) - 2
        close = self.pr.close
        ans = np.zeros(self.n)
        for i in range(level_count-1, tree.higest_leaf_lvl-1, -1):
            # print ('cl_lvl:', i)
            job = [j for j in
                   range(self.tree.level[i], self.tree.level[i+1])]
            for ind in job:
                for i_cl in close[ind]:
                    if not tree.child[ind] and not tree.child[i_cl]:
                        col_ind = self.index_lvl[ind]
                        row_ind = self.index_lvl[i_cl]
                        ans[col_ind] += self.pr.func(col_ind, row_ind).dot(tmp[row_ind])
                    elif tree.child[ind] and not tree.child[i_cl]:
                        for ch_ind in tree.child[ind]:
                            col_ind = self.index_lvl[ch_ind]
                            row_ind = self.index_lvl[i_cl]
                            ans[col_ind] += self.pr.func(col_ind, row_ind).dot(tmp[row_ind])
                    elif not tree.child[ind] and tree.child[i_cl]:
                         for ch_cl in tree.child[i_cl]:
                            col_ind = self.index_lvl[ind]
                            row_ind = self.index_lvl[ch_cl]
                            ans[col_ind] += self.pr.func(col_ind, row_ind).dot(tmp[row_ind])
        return ans
    def dot_il(self, i, tmp):
        il = self.pr.far
        ans = np.zeros(self.n)
        job = [j for j in
               range(self.tree.level[i], self.tree.level[i+1])]
        for ind in job:
            for i_il in il[ind]:
                col_ind = self.basis[ind]
                row_ind = self.basis[i_il]
                ans[col_ind] += self.pr.func(col_ind, row_ind).dot(tmp[row_ind])
        return ans
    def dot_T(self, i, tmp, prod_type):
        job = [j for j in
               range(self.tree.level[i], self.tree.level[i+1])]
        for ind in job:
            if prod_type == 'up':
                T = self.T[ind].T#np.linalg.inv(self.T[ind]).T
            else:
                T = self.T[ind]#np.linalg.inv(self.T[ind])
            tmp[self.index_lvl[ind]] = T.dot(tmp[self.index_lvl[ind]])
        return tmp
def fmm_lvl(fmm, ind_l, tau = 1e-3, l = 0):
    pr = fmm.pr
    close = pr.close
    elim_list = fmm.elim_list
    for ind in ind_l:
#   T:
        tmp = build_T(ind, fmm, tau)
        T = fmm.T[ind]
        T_T = T.T
        b = fmm.basis[ind]
        basis = fmm.local_basis[ind]

    return fmm
def build_cube_problem(func, n=15, ndim=2, block_size=28, verbose=1,point_based_tree = True, close_r='1box',num_child_tree = 'hyper', random_points=0, zk=None):
    count = n**ndim
    if random_points:
        position = np.random.rand(ndim,n**ndim)
    else:
        if ndim == 1:
            position = (np.arange(1,n+1)/(n)).reshape(1,n**ndim)
        elif ndim == 2:
            x0, x1 = np.meshgrid(np.arange(1,n+1)/(n),np.arange(1,n+1)/n)
            position = np.vstack((x0.reshape(1,n**ndim),x1.reshape(1,n**ndim)))
        elif ndim == 3:
            x0, x1, x2 = np.meshgrid(np.arange(1,n+1)/(n),np.arange(1,n+1)/n, np.arange(1,n+1)/(n))
            position = np.vstack((x0.reshape(1,n**ndim),x1.reshape(1,n**ndim), x2.reshape(1,n**ndim)))

    data = Data(ndim, count, position, close_r=close_r)
    if func == test_funcs.exp_distance_h2t:
        data.k = zk
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, verbose%2)
    return problem
def build_line(func, n=15, ndim=2, block_size=28, verbose=1,point_based_tree = True, close_r='1box',num_child_tree = 'hyper', random_points=0, zk=None):
    count = n**ndim
    if random_points:
        position = np.random.rand(ndim,n**ndim)
    else:
        if ndim == 1:
            raise NameError('In progress')
        elif ndim == 2:
            x0, x1 = np.meshgrid(np.arange(1,n+1)/(n),np.arange(1,n+1)/n)
            position = np.vstack((x0.reshape(1,n**ndim),x1.reshape(1,n**ndim)))
        elif ndim == 3:
            x0, x1, x2 = np.meshgrid(np.arange(1,n+1)/(n),np.arange(1,n+1)/n, np.arange(1,n+1)/(n))
            position = np.vstack((x0.reshape(1,n**ndim),x1.reshape(1,n**ndim), x2.reshape(1,n**ndim)))
    data = Data(ndim, count, position, close_r=close_r)
    if func == test_funcs.exp_distance_h2t:
        data.k = zk
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, verbose%2)
    return problem
def build_problem(geom_type='cube',block_size=26, n=15, ndim = 2, func = None, point_based_tree=0, close_r = 1., num_child_tree='hyper', random_points=1, file = None, eps = 0.51e-6, zk = 1.1 + 1j*0, alpha = 3.0, beta = 0, wtd_T=0,add_up_level_close=0,half_sym=0,csc_fun=0,q_fun=0,ifwrite=0, nu=10, order=10):
    iters = 2
    onfly = 1
    verbose = 0
    random_init = 2
    if point_based_tree and close_r == '1box':
        print("!!'1box' does not work with point_based_tree = 1, close_r chanjed to 1.")
        close_r = 1.
    if geom_type == 'cube':
        pr = build_cube_problem(func, n=n, ndim=ndim, block_size=block_size,
                                  verbose=verbose, point_based_tree=point_based_tree,
                                  close_r=close_r,num_child_tree = num_child_tree,random_points = random_points,zk=zk)
    elif geom_type == 'line':
        if ndim != 1:
            raise NameError (f"Geometry type '{geom_type}' works only for 1d!'")
        pr = build_line_problem(func, n=n, ndim=ndim, block_size=block_size,
                                  verbose=verbose, point_based_tree=point_based_tree,
                                  close_r=close_r,num_child_tree = num_child_tree,random_points = random_points,zk=zk)
    elif geom_type == 'sphere':
        pr = build_sphere(func, n=n, ndim=ndim, block_size=block_size,
                                   verbose=verbose, point_based_tree=point_based_tree,
                                  close_r=close_r,num_child_tree = num_child_tree,zk=zk)
    elif geom_type == 'from_file':
        if file is None:
            raise NameError(f"Geometry type '{geom_type}' should have nonepty file!")
        pr = build_problem_from_file(func, block_size=block_size, verbose=verbose,
                                     point_based_tree=point_based_tree, close_r=close_r,
                                     num_child_tree=num_child_tree, file=file,eps=eps, zk=zk, alpha=alpha,
                                     beta=beta,csc_fun=csc_fun,ifwrite=ifwrite)
    elif geom_type == 'wtorus':
        pr = build_problem_wtorus(func, block_size=block_size, verbose=verbose,
                                  point_based_tree=point_based_tree, close_r=close_r,
                                  num_child_tree=num_child_tree, eps=eps, zk=zk, alpha=alpha,
                                  beta=beta,csc_fun=csc_fun,ifwrite=ifwrite, nu=nu, order=order)
    else:
        raise NameError (f"Geometry type '{geom_type}' is not supported. Try 'cube/sphere/from_file/wtorus', !")
    pr.add_up_level_close = add_up_level_close
    if add_up_level_close:
        print('Warning! up-level close is not well-tested!')
    n_parants = 1
    for i in range(1, len(pr.tree.level)-1):
        n_nodes_lvl = pr.tree.level[i+1] - pr.tree.level[i]
        if n_nodes_lvl != n_parants * pr.tree.nchild:
            pr.tree.higest_leaf_lvl = i-1
            break
        n_parants = n_nodes_lvl
    pr.csc_fun = csc_fun
    pr.wtd_T = wtd_T
    pr.half_sym = half_sym
    pr.q_fun = q_fun
    pr.eps = eps
    tree = pr.tree
    level_count = len(tree.level) - 2
    for i in range(level_count-1, -1, -1):
        job = [j for j in
                        range(tree.level[i], tree.level[i+1])]
        exist_no_trans_t = False
        exist_no_trans_f = False
        for ind in job:
            if pr.notransition[ind]:
                exist_no_trans_t = True
            else:
                exist_no_trans_f = True
        if exist_no_trans_t and exist_no_trans_f:
            print ('lvl', i, '+-')
            pr.tail_lvl = i
            for ind in job:
                pr.notransition[ind] = False
        elif exist_no_trans_t:
            pr.tail_lvl = i+1
            break
    return pr
def build_sphere(func, n=15, ndim=3, block_size=28, verbose=1,point_based_tree = True, close_r='1box',num_child_tree = 'hyper', zk=None):
    if ndim == 2:
        raise NameError(f'ndim = 2 is in progress')
    r = 1.
    c = np.zeros(ndim)
    position = fibonacci_sphere(n, r, c)
    count = position.shape[1]
    print (f'Number of points is {count}')

    if (np.unique(position, axis=1)).shape[1] != count:
        raise NameError ('Duplocated points!')

    data = Data(ndim, count, position, close_r=close_r)
    if func == test_funcs.exp_distance_h2t:
        data.k = zk
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, verbose%2)
    return problem
def build_problem_from_file(func, block_size=28, verbose=1, point_based_tree=True, close_r='1box', num_child_tree='hyper',file=None, eps = 0.51e-6, zk = 1.1 + 1j*0, alpha = 3.0, beta = 0,csc_fun=0, ifwrite=0):
    x = np.loadtxt(file)
    ndim = 3
    order = int(x[0])
    npatches = int(x[1])
    npols = int((order+1)*(order+2)/2)
    n = npatches*npols

    zpars = np.array([zk,alpha,beta])


    # setup geometry in the correct format
    norders = order*np.ones(npatches)
    iptype = np.ones(npatches)
    srcvals = x[2::].reshape(12,n).copy(order='F')
    x0 = srcvals[0]
    x1 = srcvals[1]
    x2 = srcvals[2]
    position = np.vstack((x0.reshape(1,n),x1.reshape(1,n), x2.reshape(1,n)))

    data = Data(ndim, n, position, close_r=close_r)
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, verbose%2)
    problem.ndim = ndim
    problem.file = file
    problem.eps = eps
    return problem
def fibonacci_sphere(numpts, k, c):
    ga = (3 - np.sqrt(5)) * np.pi # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(numpts)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1/numpts-1, 1-1/numpts, numpts)

    # a list of the radii at each height step of the unit circle
    radius = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    x = radius * k * np.cos(theta) + c[0]
    y = radius * k * np.sin(theta) + c[1]
    z = z * k + c[2]
    return np.array((x, y, z))
def buildmatrix(ind, fmm, tau):
    pr = fmm.pr
    tree = fmm.tree
    index = tree.index
    ndim = tree.data.ndim
    pb_ind = []
    sch_len = 0
# proxy build points:
    p = fmm.proxy_p
    r = fmm.proxy_r
    c = np.zeros(ndim)
    for i_ndim in range(ndim):
        c[i_ndim] = (fmm.tree.aux[ind][:,i_ndim][0] + fmm.tree.aux[ind][:,i_ndim][1])/2
    box_size = np.linalg.norm(fmm.tree.aux[ind][1] - fmm.tree.aux[ind][0])
    theta = np.linspace(0, 2*np.pi, p, endpoint=False)
    if ndim == 1:
        mat = np.zeros((fmm.index_lvl[ind].size,0))
        for i_far in pr.far[ind]:
            mat_tmp = fmm.pr._func(tree.data, fmm.index_lvl[ind], tree.data, fmm.index_lvl[i_far])
            mat = np.hstack((mat,mat_tmp))
        return mat
    elif ndim == 2:
        mat = np.zeros((fmm.index_lvl[ind].size,0))
        for i_far in pr.far[ind]:
            mat_tmp = fmm.pr._func(tree.data, fmm.index_lvl[ind], tree.data, fmm.index_lvl[i_far])
            mat = np.hstack((mat,mat_tmp))
        return mat
        proxy =  r * box_size * np.vstack((c[0] + np.cos(theta), c[1] + np.sin(theta)))
        proxy[0] -= (c * r * box_size - c)[0]
        proxy[1] -= (c * r * box_size - c)[1]
    elif ndim == 3:
        proxy = fibonacci_sphere(p, r * box_size, c)
    else:
        raise NameError(f'Dimention {ndim} is in progress, we have ndim = 1,2,3')
    proxy_data = Data(tree.data.ndim, p, proxy, close_r=tree.data.close_r)
    proxy_mat = fmm.pr._func(tree.data, fmm.index_lvl[ind], proxy_data, np.arange(p))
    return proxy_mat
def build_T(ind, fmm, tau):
    pr = fmm.pr
    index_lvl = fmm.index_lvl[ind]
    matrix = buildmatrix(ind, fmm, tau)
    if matrix.shape[1] == 0:
        fmm.basis[ind] = dc(index_lvl)
        fmm.local_basis[ind] = np.arange(index_lvl.shape[0])
        fmm.T[ind] = np.identity(index_lvl.shape[0], dtype=pr.dtype)
        return 0
    k, idx, proj = interp_decomp(matrix.T, tau)
    if k == index_lvl.shape[0]:
        fmm.basis[ind] = dc(index_lvl)
        fmm.local_basis[ind] = np.arange(k)
        fmm.T[ind] = np.identity(index_lvl.shape[0], dtype=pr.dtype)
    else:
        fmm.basis[ind] = index_lvl[idx[:k]]
        fmm.local_basis[ind] = idx[:k]
        T_1 = np.identity(index_lvl.shape[0], dtype=pr.dtype)
        if proj.T.shape[0] > 0:
            T_1[np.ix_(idx[k:],idx[:k])] = proj.T# * -1
        fmm.T[ind] = T_1
    return 0
def build_fmm(pr, proxy_p=10, proxy_r=1.):
    warnings.warn("In this wersion in 1d and 2d low-rank is computed for interaction list, not for proxy")
    tree = pr.tree
    level_count = len(tree.level) - 2

    fmm = Fmm(pr, proxy_p=proxy_p, proxy_r=proxy_r, symmetric_fun = 1)
    ind_l = []
    fmm.tail_lvl = pr.tail_lvl

    for i in range(level_count-1, fmm.tail_lvl-1, -1):
        job = [j for j in
               range(tree.level[i], tree.level[i+1])]
        for ind in job:
            if tree.child[ind]:
                fmm.upd_index_lvl(ind)
            else:
                fmm.index_lvl[ind] = fmm.tree.index[ind]
        ind_l += job
        fmm = fmm_lvl(fmm, job, tau=pr.eps, l=i)
    return fmm
