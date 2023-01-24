import numpy as np


def project(x, A):
    """
    Projection of point x on the subspace defined by points given in A (as row).
    """
    if A.shape[1] == 1:
        return A[:, [0]]
    else:
        yp, y, A = A[:, 0], A[:, [0]], A[:, 1:]
        for i in range(A.shape[1]):
            A[:, i] -= yp
        # print(A)
        Q = np.linalg.qr(A)[0]
        return x - (Q @ (Q.T @ (x - y)) + y)


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
    support_points = np.array([S(_) for _ in simplex_vertices])
    # Step 1.3: calculate the center of the resulting simplex
    x = np.average(support_points, axis=0).reshape((n, 1))
    # Phase 2: performing GJK algorithm
    d = np.copy(x)
    s = np.copy(x)
    k = 0
    replaced = False
    W = np.empty((n, 0))
    while True:
        delta = (k+1)/(k+3)
        if not replaced:
            y = delta*x + (1-delta)*s
            d = delta*d/np.linalg.norm(d) + (1-delta) * \
                (y - z)/np.linalg.norm(y - z)
        else:
            d = x
        # print(replaced)
        # print("x =", x)
        # print("d =", d)
        s = S(d)
        # print("s =", s)
        if np.dot(x.T, x - s)[0, 0] <= eps:
            if np.linalg.norm(d - x) <= eps:
                break
            s = S(x - z)
            replaced = True
        # Reduce the set W u {s} to the minimal set of vertices whose convex hull contains the projection of z
        witness = np.hstack((W, s))
        # print("Witness =", witness)
        if witness.shape[1] == n+1:
            W = np.empty((n, 0))
            for i in range(witness.shape[1]):
                m = witness[:, i]
                A = np.hstack((witness[:, :i], witness[:, i+1:]))
                # Projection of z and m onto the hyperplane
                z_p = z - project(z, A)
                m_p = m - project(m, A)
                # print(m)
                # print(m_p)
                # print(z)
                # print(z_p)
                if np.dot(m_p.T, z_p)[0, 0] > 0:
                    # z and m lie on the same half-space, so we include point m in our new set W
                    W = np.hstack((W, m))
        else:
            W = witness
        # QR decomposition to find the orthonormal basis of those vertices
        # print("W =", W)
        x = project(z, W)
        # print(x)
        # print(z)
        # print(np.linalg.norm(x-z))
        # If z lies inside the simplex, we have found our witness simplex
        if np.linalg.norm(x - z) <= eps:
            break
        k += 1
    # if the algorithm terminates without returning, then z must lie on the boundary of the witness simplex.
    # We then find a facet containing z, and check if it is a facet of the convex hull of the polytope.
    # print("Final witness =", W)
    if W.shape[1] == 1:
        return True
    else:
        for i in range(W.shape[1]):
            # Projection of z onto the hyperplane
            A = np.hstack((witness[:, :i], witness[:, i+1:]))
            z_p = z - project(z, A)
            if np.linalg.norm(z_p) <= eps:
                # this facet contains z
                m_p = m - project(m, A)
                t = S(m_p)
                t_p = t - project(t, A)
                if np.linalg.norm(t_p) < eps:
                    return True
    return False


def s(x):
    if np.linalg.norm(x) == 0:
        return sample_spherical(d)
    return -x/np.linalg.norm(x)


def sample_spherical(ndim):
    vec = np.random.randn(ndim, 1)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


d = 7
N = 100000
for _ in range(N):
    x = sample_spherical(d)
    # x = np.array([[0.83711192], [-0.54703164]])
    if not inside_check(s, x):
        print("Error 1!")
        print(x)
        print(np.linalg.norm(x))
for _ in range(N):
    x = sample_spherical(d)*(1 - 5e-8)
    # x = np.array([[0.83711192], [-0.54703164]])
    if inside_check(s, x):
        print("Error 2!")
        print(x)
        print(np.linalg.norm(x))
