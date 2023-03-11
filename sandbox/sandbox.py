from sage.all import *
import numpy as np
import time


class PolytopeAlgebra:
    def __init__(self, polytope=None, dimension=2, base_ring=ZZ):
        # self.D = D if D is not None else None
        self.dimension = dimension
        if polytope is None:
            self.polytope = Polyhedron(
                ambient_dim=dimension, base_ring=base_ring)
        else:
            self.polytope = Polyhedron(vertices=polytope, base_ring=base_ring)

    def __mul__(self, other):
        if self.polytope.dim() == -1:
            return other
        elif other.polytope.dim() == -1:
            return self
        else:
            result = PolytopeAlgebra(dimension=self.dimension)
            result.polytope = self.polytope + other.polytope
            return result

    def __add__(self, other):
        result = PolytopeAlgebra(dimension=self.dimension)
        result.polytope = self.polytope.convex_hull(other.polytope)
        return result


def nussinov(rna):
    n = len(rna)
    d = 3
    N = np.array([[PolytopeAlgebra(dimension=d)
                 for _ in range(n)] for __ in range(n)])
    for d in range(1, n):
        for i in range(0, n-d):
            j = i + d
            if rna[i] + rna[j] in ["AU", "UA"]:
                N[i, j] = N[i+1, j-1] * \
                    PolytopeAlgebra([[1, 0, 0]])
            elif rna[i] + rna[j] in ["GC", "CG"]:
                N[i, j] = N[i+1, j-1] * \
                    PolytopeAlgebra([[0, 1, 0]])
            elif rna[i] + rna[j] in ["GU", "UG"]:
                N[i, j] = N[i+1, j-1] * \
                    PolytopeAlgebra([[0, 0, 1]])
            for l in range(i, j):
                N[i, j] += N[i, l] * N[l+1, j]
    return N[0, n-1]


seed = 472
n = 150
rng = np.random.default_rng(seed)
rna = ''.join(rng.choice(['A', 'G', 'C', 'U']) for _ in range(n))
print(rna)
tic = time.time()
P = nussinov(rna).polytope
toc = time.time()
print(toc - tic)

r = [f.ambient_V_indices() for f in P.facets()]
print(P)
print(r)
