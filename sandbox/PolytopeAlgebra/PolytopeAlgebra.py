import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import itertools
import random


class PolytopeAlgebra:
    def __init__(self, dimension=2, points=None, D=lambda _: True):
        self.D = D if D is not None else None
        self.points = points if points is not None else np.empty(
            (0, dimension))
        self.dimension = dimension

    def __add__(self, other):
        union = np.vstack((self.points, other.points))
        if union.shape[0] <= self.dimension:
            return PolytopeAlgebra(self.dimension)
        else:
            convex_hull = ConvexHull(union)
            return PolytopeAlgebra(self.dimension, union[convex_hull])

    def __mul__(self, other):
        if self.points.shape[0] == 0:
            return PolytopeAlgebra(self.dimension, other.points)
        elif other.points.shape[0] == 0:
            return PolytopeAlgebra(self.dimension, self.points)
        else:
            minkowski_sum = np.sum(
                np.array(list(itertools.product(self.points, other.points))), axis=1)
            if minkowski_sum.shape[0] <= self.dimension:
                return PolytopeAlgebra(self.dimension, minkowski_sum)
            else:
                convex_hull = ConvexHull(minkowski_sum)
                return PolytopeAlgebra(self.dimension, minkowski_sum[convex_hull])

def test(n):
    rna = ''.join(random.choices("ACGU", k=n))
    d = 3
    tic = time.time()
    N = np.array([[PolytopeAlgebra(d) for _ in range(n)] for __ in range(n)])
    for i in range(n):
        for j in range(i+1, n):
            for l in range(i, j):
                N[i, j] += N[i, l] * N[l+1, j]
            if rna[i] + rna[j] in ["AU", "UA"]:
                N[i, j] *= PolytopeAlgebra(d, np.array([1, 0, 0]))
            if rna[i] + rna[j] in ["GC", "CG"]:
                N[i, j] *= PolytopeAlgebra(d, np.array([0, 1, 0]))
            if rna[i] + rna[j] in ["GU", "UG"]:
                N[i, j] *= PolytopeAlgebra(d, np.array([0, 0, 1]))
    toc = time.time()
    return toc-tic


print(test(1000))
