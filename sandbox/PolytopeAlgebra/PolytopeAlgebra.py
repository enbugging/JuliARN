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


def test(n):
    rna = ''.join(random.choice("ACGU") for _ in range(n))
    d = 3
    tic = time.time()
    N = np.array([[PolytopeAlgebra(d) for _ in range(n)] for __ in range(n)])
    for d in range(1, n):
        for i in range(0, n-d):
            j = i + d
            if rna[i] + rna[j] in ["AU", "UA"]:
                N[i, j] = N[i+1, j-1] * \
                    PolytopeAlgebra(d, np.array([[1, 0, 0]]))
            elif rna[i] + rna[j] in ["GC", "CG"]:
                N[i, j] = N[i+1, j-1] * \
                    PolytopeAlgebra(d, np.array([[0, 1, 0]]))
            elif rna[i] + rna[j] in ["GU", "UG"]:
                N[i, j] = N[i+1, j-1] * \
                    PolytopeAlgebra(d, np.array([[0, 0, 1]]))
            for l in range(i, j):
                N[i, j] += N[i, l] * N[l+1, j]
    toc = time.time()
    return toc-tic


#print(N[0, n-1].points)
print(test(100))
