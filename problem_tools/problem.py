import numpy as np
from copy import deepcopy as dc
from collections import defaultdict
from itertools import product
from numba import jit
class Problem(object):
    def __init__(self, func, tree,  verbose=False):
        self._func = func
        # self.symmetric = symmetric
        self.tree = tree
        self.data = tree.data
        self.shape = (len(tree.data), len(tree.data))
        l = np.arange(1, dtype=np.uint64)
        tmp = self.func(l, l)
        self.func_shape = tmp.shape[1:-1]
        self.dtype = tmp.dtype
        self._build(verbose)

    def _build(self, verbose=False):
        row_check = [[0]]
        self.far = []
        self.close = []
        self.notransition = []
        self.tree.aux =[self.data.compute_aux(self.tree.index[0])]
        cur_level = 0
        col_tree = self.tree
        while (self.tree.level[cur_level] < self.tree.level[cur_level+1]):
            for i in range(self.tree.level[cur_level],self.tree.level[cur_level+1]):
                self.far.append([])
                self.close.append([])
            for i in range(self.tree.level[cur_level],self.tree.level[cur_level+1]):
                for j in row_check[i]:
                    if self.tree.is_far(i, col_tree, j):
                        self.far[i].append(j)

                    else:
                        self.close[i].append(j)
            for i in range(self.tree.level[cur_level],self.tree.level[cur_level+1]):
                if i == 0:
                    self.notransition.append(not self.far[i])
                else:
                    self.notransition.append(not(self.far[i] or
                        not self.notransition[self.tree.parent[i]]))
            for i in range(self.tree.level[cur_level],self.tree.level[cur_level+1]):
                if (self.close[i] and not self.tree.child[i] and
                        self.tree.index[i].size >
                        self.tree.block_size):
                    nonzero_close = False
                    for j in self.close[i]:
                        if (col_tree.index[j].size >
                                col_tree.block_size):
                            nonzero_close = True
                            break
                    if nonzero_close:
                        self.tree.divide(i)
            for i in range(self.tree.level[cur_level],self.tree.level[cur_level+1]):
                whom_to_check = []
                for j in self.close[i]:
                    whom_to_check.extend(col_tree.child[j])
                for j in self.tree.child[i]:
                    row_check.append(whom_to_check)
            self.tree.level.append(len(self.tree))
            cur_level += 1
        self.num_levels = len(self.tree.level)-1
        self.tree.num_levels = self.num_levels
    def func(self, row, col):
        return self._func(self.data, row, self.data, col)
