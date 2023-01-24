import numpy as np
from scipy.spatial import ConvexHull


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


def inside_check(S, z, eps=1e-9):
    """
    Implementation of Nesterov-accelerated GJK algorithm, return True if z
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
    support_points = np.array([S(_.reshape(n, 1) + z).reshape(n)
                              for _ in simplex_vertices])
    # Step 1.3: calculate the center of the resulting simplex
    x = np.average(support_points, axis=0).reshape((n, 1)) - z
    # Phase 2: performing GJK algorithm
    d = np.copy(x)
    s = np.copy(x)
    k = 0
    replaced = False
    W = np.empty((n, 0))
    if np.linalg.norm(x) <= eps:
        return True
    while True:
        delta = (k+1)/(k+3)
        if not replaced:
            y = delta*x + (1-delta)*s
            d = delta*d/np.linalg.norm(d) + (1-delta) * y/np.linalg.norm(y)
        else:
            d = x
        # print(replaced)
        # print("x =", x)
        # print("d =", d)
        s = S(d) - z
        # print("s =", s)
        # print(W)
        if np.dot(x.T, x - s)[0, 0] <= eps**2:
            if np.linalg.norm(d - x) <= eps:
                return np.linalg.norm(x) <= eps
            s = S(x) - z
            replaced = True
        # Reduce the set W u {s} to the minimal set of vertices whose convex hull contains the projection of z
        witness = np.unique(np.hstack((W, s + z)), axis=1)
        if witness.shape[1] == W.shape[1]:
            k += 1
            continue
        # print("Witness =", witness)
        if witness.shape[1] == n+1:
            W = np.empty((n, 0))
            for i in range(witness.shape[1]):
                m = witness[:, [i]]
                A = np.hstack((witness[:, :i], witness[:, i+1:]))
                # Projection of z and m onto the hyperplane
                z_p = z - project(z, A)
                m_p = m - project(m, A)
                # print(m)
                # print(m_p)
                # print(z)
                # print(z_p)
                # print("As dot product =", np.dot(m_p.T, z_p)[0, 0])
                if np.dot(m_p.T, z_p)[0, 0] > 0:
                    # z and m lie on the same half-space, so we include point m in our new set W
                    W = np.hstack((W, m))
                    # print("Pick ", m)
        else:
            W = witness
        # QR decomposition to find the orthonormal basis of those vertices
        # print("W =", W)
        x = project(z, W) - z
        # print("W =", W)
        # If z lies inside the simplex, we have found our witness simplex
        if np.linalg.norm(x) <= eps:
            break
        k += 1

    # if the algorithm terminates without returning, then z must lie on the boundary of the witness simplex.
    # We then find a facet containing z, and check if it is a facet of the convex hull of the polytope.
    # print("Final witness =", W)
    if W.shape[1] == 1:
        return True
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


def test_sphere(d=7, N=100000, seed=None):
    rng = np.random.default_rng(seed)

    def s(x):
        if np.linalg.norm(x) == 0:
            return sample_spherical(d)
        return -x/np.linalg.norm(x)

    def sample_spherical(ndim):
        vec = rng.standard_normal((ndim, 1))
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    for _ in range(N):
        x = sample_spherical(d)
        if not inside_check(s, x):
            print("Error 1!")
            print(x)
            print(np.linalg.norm(x))
    for _ in range(0):
        x = sample_spherical(d)*(1 - 5e-8)
        if inside_check(s, x):
            print("Error 2!")
            print(x)
            print(np.linalg.norm(x))


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
        P = generate_convex_polytope()
        z = P[rng.integers(P.shape[0])].reshape(d, 1)
        if not inside_check(lambda x: support(x, P), z):
            print("Error 1!")
            print(z)
            print(P)

    for _ in range(N):
        P = generate_convex_polytope()
        coeff = rng.random((P.shape[0], 1))
        coeff /= np.sum(coeff)
        coeff *= 1 - 5e-8
        z = P.T @ coeff
        if inside_check(lambda x: support(x, P), z):
            print("Error 2!")
            print(z)
            print(P)


# test_sphere(d=2, N=1, seed=17)
test_convex_polytope(d=7, N=10000, npoints=10000)
