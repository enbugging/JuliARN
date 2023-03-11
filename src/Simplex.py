import numpy as np


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


def farthest_hyperplane_pivot_rule(T, x):
    """
    Farthest Hyperplane pivot rule: in R^d, given a simplex of d+1 vertices
    and a point x, we consider all combinations of d vertices that form a 
    hyperplane separating x and the remaining vertex.

    If such a hyperplane exists, this rule returns a d*d matrix, where the 
    columns are the vertices that constitute the hyperplane that is farthest 
    from x.

    Else, it returns the whole simplex.

    T: a np.array of shape (d, d+1), where the columns are the vertices of simplex.
    x: our point of interest
    """
    max_dist = -1
    for i in range(T.shape[1]):
        # a d*d matrix denoting d vertices
        A = np.hstack((T[:, :i], T[:, i+1:]))
        m = T[:, [i]]  # m is the remaining vertex
        # Projection of x and m onto the hyperplane
        x_p = x - project(x, A)
        m_p = m - project(m, A)
        if np.dot(m_p.T, x_p)[0, 0] < 0:
            # meaning the hyperplane defined by d vertices in A separates x and m
            if max_dist < np.linalg.norm(x_p):
                max_dist = np.linalg.norm(x_p)
                W = np.copy(A)
    if max_dist == -1:
        # no hyperplane separates x and m, i.e. x is inside the simplex
        return T
    else:
        return W


def simplexOnBoundary(S, x, eps=1e-9, max_iter=10, pviot_rule=farthest_hyperplane_pivot_rule):
    """
    Implementation of GJK algorithm, return True if z
    is on the boundary of the convex hull defined by S. Here we assume z
    does not lie outside the convex hull.
    s: support function
    x: point to check
    """
    d = x.size
    v = np.zeros((d, 1))
    v[0, 0] = 1
    W = np.empty((d, 0))

    # Phase 1: we build the first d vertice of the simplex
    for _ in range(d):
        w = S(v)
        if np.linalg.norm(w - x) <= eps:
            return True
        W = np.hstack((W, w))
        v = x - project(x, W)
    # Phase 2: we refine the simplex. Applicable iff W has d vertices already.
    if W.shape[1] == d:
        for _ in range(d, max_iter):
            w = S(v)
            if np.linalg.norm(w - x) <= eps:
                return True
            T = np.unique(np.hstack((W, w)), axis=1)
            W = pviot_rule(T, x)
            if W.shape[1] == d+1:
                # meaning we have found our witness simplex
                break
            v = x - project(x, W)

    # Here, we have found a simplex that contains x
    if W.shape[1] == 1:
        # We got lucky on the first iteration, and the simplex is just a point
        # We still check for numerical stablity
        return np.linalg.norm(W[:, 0] - x) <= eps
    else:
        # We check if x is on the boundary of the simplex
        for i in range(W.shape[1]):
            m = W[:, [i]]
            # The stronger is true: x is one of the vertices!
            if np.linalg.norm(m - x) <= eps:
                return True

            # Projection of x onto the hyperplane
            A = np.hstack((W[:, :i], W[:, i+1:]))
            x_p = x - project(x, A)

            if np.linalg.norm(x_p) <= eps:
                # this facet contains x
                m_p = m - project(m, A)
                t = S(m_p)
                t_p = t - project(t, A)
                if np.linalg.norm(t_p) <= eps:
                    return True
        return False


def passthrough(x, o, P):
    """
    Determine if the segment ox intersects the "portal" polytope P.
    """
    A = np.copy(P)
    for i in range(1, A.shape[1]):
        A[:, i] -= A[:, 0]
    A[:, [0]] = x - o
    #print(P, A, x, o, sep='\d')
    sol = np.linalg.solve(A, x - P[:, [0]])
    return sol[0, 0] <= 1 and np.sum(sol[1:, 0]) <= 1 and np.all(sol >= 0)


def mprOnBoundary(S, x, eps=1e-9, max_iter=100):
    """
    Implementation of Minkowski Portal Refinement algorithm, return True if z
    is on the boundary of the convex hull defined by S. Here we assume z
    does not lie outside the convex hull.

    Here we assume that the polytope is non-degenerate.
    s: support function
    x: point to check
    """
    d = x.size
    v = np.zeros((d, 1))
    v[0, 0] = 1

    # Phase 1: Determine an initial portal and a point in the interior
    W = S(v)
    if np.linalg.norm(W - x) <= eps:
        return True
    for _ in range(d):
        v = x - project(x, W)
        w = S(v)
        if np.linalg.norm(w - x) <= eps:
            return True
        W = np.hstack((W, w))
    origin = np.average(W, axis=1).reshape(d, 1)
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
    for _ in range(d, max_iter):
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

    # If we reach this stage, we should have a portal that contains x
    if W.shape[1] == 1:
        # We got lucky on the first iteration, and the portal is just a point
        return np.linalg.norm(W[:, 0] - x) <= eps
    else:
        for i in range(W.shape[1]):
            m = W[:, [i]]
            if np.linalg.norm(m - x) <= eps:
                return True
            # Projection of x onto the hyperplane
            A = np.hstack((W[:, :i], W[:, i+1:]))
            x_p = x - project(x, A)
            if np.linalg.norm(x_p) <= eps:
                # this facet contains x
                m_p = m - project(m, A)
                t = S(m_p)
                t_p = t - project(t, A)
                if np.linalg.norm(t_p) <= eps:
                    return True
        return False
