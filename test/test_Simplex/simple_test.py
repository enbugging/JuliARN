from scipy.spatial import ConvexHull
import numpy as np

import sys
sys.path.append("../../")
from src import Simplex as sp

def test_convex_polytope(onBoundary, d=2, N=100, npoints=100, seed=None):
    rng = np.random.default_rng(seed)

    def generate_convex_polytope():
        points = rng.random((npoints, d))
        hull = ConvexHull(points)
        return points[hull.vertices]  # *100

    def support(x, P):
        dotprod = ((P @ x).T)[0]
        min_points = P[np.where(dotprod == dotprod.max())].T
        coeff = rng.random((min_points.shape[1], 1))
        coeff /= np.sum(coeff)
        return min_points @ coeff
    for _ in range(N):
        P = generate_convex_polytope()
        z = P[rng.integers(P.shape[0])].reshape(d, 1)
        if not onBoundary(lambda x: support(x, P), z):
            print("Error 1!")
            print(z)
            print(P)

    for _ in range(1):
        P = generate_convex_polytope()
        coeff = rng.random((P.shape[0], 1))
        coeff /= np.sum(coeff)
        # coeff *= 1 - 5e-8
        z = P.T @ coeff
        if onBoundary(lambda x: support(x, P), z):
            print("Error 2!")
            print(z)
            print(P)


for s in range(2000, 3000):
    if s % 100 == 0:
        print("Seed =", s)
    test_convex_polytope(sp.mprOnBoundary, d=3,
                         N=1, npoints=1000, seed=s)
