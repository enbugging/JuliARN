import numpy as np
from scipy.spatial import ConvexHull
import time


def project(x, A):
    """
    Projection of point x on the subspace defined by points given in A (as row).
    """
    if A.shape[1] == 1:
        return A[:, [0]]
    else:
        yp, y = A[:, 0], A[:, [0]]
        B = np.copy(A[:, 1:])
        for i in range(B.shape[1]):
            B[:, i] -= yp
        Q = np.linalg.qr(B)[0]
        return Q @ (Q.T @ (x - y)) + y


def inside_check(S, z, eps=1e-9, max_iter=100):
    """
    Implementation of GJK algorithm, return True if z
    is on the boundary of the convex hull defined by S. Here we assume z
    does not lie outside the convex hull.
    s: support function
    z: point to check
    """
    # Phase 1: find a point inside the convex hull
    # Step 1.1: get n+1 points that forms a regular n-simplex
    # Trick from https://mathoverflow.net/questions/38724/coordinates-of-vertices-of-regular-simplex
    n = z.size
    simplex_vertices = np.linalg.qr(
        np.ones((n+1, 1)), mode='complete')[0][:, 1:]
    # Step 1.2: get support points corresponding to vectors of the regular n-simplex
    support_points = np.array([S(_.reshape(n, 1)).reshape(n)
                              for _ in simplex_vertices])
    # Step 1.3: calculate the center of the resulting simplex
    x = np.average(support_points, axis=0).reshape(n, 1) - z
    # Phase 2: performing GJK algorithm
    W = np.empty((n, 0))
    if np.linalg.norm(x) <= eps:
        return True
    for k in range(max_iter):
        # print("K = ", k)
        s = S(x) - z
        if np.linalg.norm(s) <= eps:
            return True
        if 2 * x.T @ (x - s) <= eps:
            break
        witness = np.unique(np.hstack((W, s + z)), axis=1)
        # print("W =", W)
        # print("s =", s)
        # print("witness =", witness)
        if witness.shape[1] == n+1:
            t = project(z, witness)
            W = np.empty((n, 0))
            for i in range(witness.shape[1]):
                m = witness[:, [i]]
                A = np.hstack((witness[:, :i], witness[:, i+1:]))
                # Projection of z and m onto the hyperplane
                t_p = t - project(t, A)
                m_p = m - project(m, A)
                # print(m)
                # print(m_p)
                # print(t)
                # print(t_p)
                # print("As dot product =", np.dot(m_p.T, t_p)[0, 0])
                if np.dot(m_p.T, t_p)[0, 0] < 0:
                    W = np.copy(A)
                    break
            if W.shape[1] == 0:
                # meaning we have found our witness simplex
                return False
        else:
            W = np.copy(witness)
        x = project(z, W) - z
        # print("W =", W)
        # print("x =", x)
        # print(project(z, W))
        # print(project(z, W) - z)
        # print(np.linalg.norm(x))
        if np.linalg.norm(x) <= eps:
            break

    # if the algorithm terminates without returning, then z must lie on the boundary of the witness simplex.
    # We then find a facet containing z, and check if it is a facet of the convex hull of the polytope.
    # print("Final witness =", W)
    if W.shape[1] == 1:
        return np.linalg.norm(W[:, 0] - z) <= eps
    else:
        for i in range(W.shape[1]):
            m = W[:, [i]]
            if np.linalg.norm(m - z) <= eps:
                return True
            # Projection of z onto the hyperplane
            A = np.hstack((witness[:, :i], witness[:, i+1:]))
            z_p = z - project(z, A)
            if np.linalg.norm(z_p) <= eps:
                # this facet contains z
                m_p = m - project(m, A)
                t = S(m_p)
                t_p = t - project(t, A)
                if np.linalg.norm(t_p) <= eps:
                    return True
        return False


def test_convex_polytope(d=2, N=100, npoints=100, seed=None):
    rng = np.random.default_rng(seed)

    def generate_convex_polytope():
        points = rng.random((npoints, d))
        hull = ConvexHull(points)
        return points[hull.vertices]*100

    def support(x, P):
        dotprod = ((P @ x).T)[0]
        min_points = P[np.where(dotprod == dotprod.min())].T
        coeff = rng.random((min_points.shape[1], 1))
        coeff /= np.sum(coeff)
        return min_points @ coeff
    for _ in range(N):
        # print("Phase 1")
        P = generate_convex_polytope()
        z = P[rng.integers(P.shape[0])].reshape(d, 1)
        # print("Phase 2")
        if not inside_check(lambda x: support(x, P), z):
            print("Error 1!")
            print(z)
            print(P)

    for _ in range(N):
        # print("Phase 3")
        P = generate_convex_polytope()
        coeff = rng.random((P.shape[0], 1))
        coeff /= np.sum(coeff)
        coeff *= 1 - 5e-8
        z = P.T @ coeff
        # print("Phase 4")
        if inside_check(lambda x: support(x, P), z):
            print("Error 2!")
            print(z)
            print(P)


# test_sphere(d=2, N=1, seed=17)
# start = time.time()
for _ in range(100):
    s = np.random.randint(1000, 2000)
    print("Seed =", s)
    test_convex_polytope(d=5, N=1, npoints=10000, seed=s)
# end = time.time()
# print(end - start)
