import numpy as np
from scipy.spatial import ConvexHull
import time
import scipy.stats as st

max_k_minkow = 0
max_k_gjk = 0


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


def passthrough(x, o, P):
    """
    Determine if the segment ox intersects the "portal" polytope P.
    """
    A = np.copy(P)
    for i in range(1, A.shape[1]):
        A[:, i] -= A[:, 0]
    A[:, [0]] = x - o
    #print(P, A, x, o, sep='\n')
    sol = np.linalg.solve(A, x - P[:, [0]])
    return sol[0, 0] <= 1 and np.sum(sol[1:, 0]) <= 1 and np.all(sol >= 0)


def inside_check_minkow(S, x, eps=1e-9, max_iter=100):
    """
    Implementation of Minkowski Portal Refinement algorithm, return True if z
    is on the boundary of the convex hull defined by S. Here we assume z
    does not lie outside the convex hull.

    Here we assume that the polytope is non-degenerate.
    s: support function
    x: point to check
    """
    n = x.size
    v = np.zeros((n, 1))
    v[0, 0] = 1
    # Phase 1: Determine an initial portal and a point in the interior
    W = S(v)
    if np.linalg.norm(W - x) <= eps:
        return True
    for k in range(1, n+1):
        global max_k_minkow
        max_k_minkow = max(max_k_minkow, k)
        v = x - project(x, W)
        w = S(v)
        if np.linalg.norm(w - x) <= eps:
            return True
        W = np.hstack((W, w))
    origin = np.average(W, axis=1).reshape(n, 1)
    found_portal = False
    for i in range(W.shape[1]):
        A = np.hstack((W[:, :i], W[:, i+1:]))
        if passthrough(x, origin, A):
            found_portal = True
            W = np.copy(A)
            break
    if not found_portal:
        # meaning x is inside W, so W is a witness simplex
        return False
    # Phase 2: Refine the portal
    for k in range(n+1, max_iter):
        max_k_minkow = max(max_k_minkow, k)
        v = x - project(x, W)
        w = S(v)
        if np.linalg.norm(w - x) < eps:
            return True
        W = np.hstack((W, w))
        found_portal = False
        for i in range(W.shape[1]-1):
            A = np.hstack((W[:, :i], W[:, i+1:]))
            if passthrough(x, origin, A):
                found_portal = True
                W = np.copy(A)
                break
        if not found_portal:
            break
    if W.shape[1] == 1:
        return np.linalg.norm(W[:, 0] - x) <= eps
    else:
        for i in range(W.shape[1]):
            m = W[:, [i]]
            if np.linalg.norm(m - x) <= eps:
                return True
            # Projection of x onto the hyperplane
            A = np.hstack((W[:, :i], W[:, i+1:]))
            z_p = x - project(x, A)
            if np.linalg.norm(z_p) <= eps:
                # this facet contains x
                m_p = m - project(m, A)
                t = S(m_p)
                t_p = t - project(t, A)
                if np.linalg.norm(t_p) <= eps:
                    return True
        return False


def inside_check_gjk(S, x, eps=1e-9, max_iter=100):
    """
    Implementation of GJK algorithm, return True if z
    is on the boundary of the convex hull defined by S. Here we assume z
    does not lie outside the convex hull.
    s: support function
    x: point to check
    """
    n = x.size
    v = np.zeros((n, 1))
    v[0, 0] = 1
    W = np.empty((n, 0))
    for k in range(max_iter):
        global max_k_gjk
        max_k_gjk = max(max_k_gjk, k)
        # print("K =", k)
        # print("W =", W)
        w = S(v) - x
        # print("v =", v)
        # print("w =", w)
        # print(v.T @ (w - (1 - eps**2) * v))
        if (W.shape[1] and np.linalg.norm(w - project(w, W)) <= eps):  # \
            # or v.T @ (w - (1 - eps**2) * v) <= 0:
            break
        T = np.unique(np.hstack((W, x + w)), axis=1)
        # print("T =", T)
        if T.shape[1] == n+1:
            t = project(x, T)
            # print("t =", t)
            W = np.empty((n, 0))
            # possible_index = []
            max_norm = 0
            for i in range(T.shape[1]):
                m = T[:, [i]]
                A = np.hstack((T[:, :i], T[:, i+1:]))
                # Projection of z and m onto the hyperplane
                t_p = t - project(t, A)
                m_p = m - project(m, A)
                # print("Inner product: ", np.dot(m_p.T, t_p)[0, 0])
                if np.dot(m_p.T, t_p)[0, 0] < 0:  # >= -eps:
                    # possible_index.append(i)
                    if np.linalg.norm(t_p) > max_norm:
                        max_norm = np.linalg.norm(t_p)
                        W = np.copy(A)
                    #W = np.hstack((W, m))
                    # print("Pick ", m)
            # if possible_index:
            #    i = np.random.choice(possible_index)
            #    W = np.hstack((T[:, :i], T[:, i+1:]))
            #W = np.unique(W, axis=1)
        else:
            W = np.copy(T)
        # print(W)
        # print(np.linalg.norm(v)**2)
        if W.shape[1] == 0:
            # meaning we have found our witness simplex
            W = np.copy(T)
            break
        v = project(x, W) - x
        # print("Norm:", np.linalg.norm(v))
        if np.linalg.norm(v) <= eps:
            return True
    if W.shape[1] == 1:
        return np.linalg.norm(W[:, 0] - x) <= eps
    else:
        for i in range(W.shape[1]):
            m = W[:, [i]]
            if np.linalg.norm(m - x) <= eps:
                return True
            # Projection of x onto the hyperplane
            A = np.hstack((W[:, :i], W[:, i+1:]))
            z_p = x - project(x, A)
            if np.linalg.norm(z_p) <= eps:
                # this facet contains x
                m_p = m - project(m, A)
                t = S(m_p)
                t_p = t - project(t, A)
                if np.linalg.norm(t_p) <= eps:
                    return True
        return False


def test_convex_polytope(d=2, N=100, npoints=100, seed=None):
    global max_k_minkow
    global max_k_minkow_s
    global max_k_gjk
    global max_k_gjk_s
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
        max_k_gjk = 0
        max_k_minkow = 0
        if not inside_check_gjk(lambda x: support(-x, P), z):
            print("GJK failed 1 with seed", seed)
        if not inside_check_minkow(lambda x: support(x, P), z):
            print("Minkow failed 1 with seed", seed)
        max_k_gjk_s.append(max_k_gjk)
        max_k_minkow_s.append(max_k_minkow)

    for _ in range(1):
        P = generate_convex_polytope()
        coeff = rng.random((P.shape[0], 1))
        coeff /= np.sum(coeff)
        # coeff *= 1 - 5e-8
        z = P.T @ coeff
        max_k_gjk = 0
        max_k_minkow = 0
        if inside_check_gjk(lambda x: support(-x, P), z):
            print("GJK failed 2 with seed", seed)
        if inside_check_minkow(lambda x: support(x, P), z):
            print("Minkow failed 2 with seed", seed)
        max_k_gjk_s.append(max_k_gjk)
        max_k_minkow_s.append(max_k_minkow)


# test_sphere(d=2, N=1, seed=17)
start = time.time()
max_k_minkow_s = []
max_k_gjk_s = []
for _ in range(1000):
    #s = np.random.randint(1000, 2000)
    s = _ + 2000
    #print("Seed =", s)
    if _ % 100 == 0:
        print("Iteration", _)
    test_convex_polytope(d=6, N=1, npoints=1000, seed=s)
max_k_gjk_s = np.array(max_k_gjk_s)
max_k_minkow_s = np.array(max_k_minkow_s)
max_k_s_better = max_k_gjk_s < max_k_minkow_s
max_k_s_as_good = max_k_gjk_s == max_k_minkow_s
max_k_s_worse = max_k_gjk_s > max_k_minkow_s
print(np.mean(max_k_s_better))
print(np.mean(max_k_s_as_good))
print(np.mean(max_k_s_worse))
delta_better = max_k_gjk_s - max_k_minkow_s
delta_worse = max_k_minkow_s - max_k_gjk_s
delta_better = delta_better[delta_better > 0]
delta_worse = delta_worse[delta_worse > 0]
print(np.mean(delta_better))
print(np.mean(delta_worse))
#max_k_s = max_k_minkow_s - max_k_gjk_s
# print(np.mean(max_k_s))
# print(np.var(max_k_s))

# print(np.mean(max_k_gjk_s))
# print(st.t.interval(0.95, len(max_k_gjk_s)-1,
#      loc=np.mean(max_k_gjk_s), scale=st.sem(max_k_gjk_s)))
# print(np.mean(max_k_minkow_s))
# print(st.t.interval(0.95, len(max_k_minkow_s)-1,
#      loc=np.mean(max_k_minkow_s), scale=st.sem(max_k_minkow_s)))
# print(np.max(max_k_gjk_s))
# print(np.max(max_k_minkow_s))
# test_convex_polytope(d=5, N=1, npoints=100, seed=1539)
# Seed to kill the improved GJK: d=3, N=1, npoints=100, seed=1294 and 1552
end = time.time()
print(end - start)
# 1825
