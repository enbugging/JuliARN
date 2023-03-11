import numpy as np


def nussinov(x: str, p: np.ndarray):
    """
    Given a string denoting a RNA sequence, return the optimal
    secondary structure without pseudoknots following Nussinov's
    algorithm. The result is denoted by a dot-bracket string.

    Input:
        x : a string denoting the RNA nucleotide sequence.
        p : a ndarray of shape (3, 1), denoting the cost parameters.
        The three entries respectively denote
        - the desirability of an A-U or U-A base pair,
        - the desirability of  a C-G or G-C base pair,
        - the desirability of an U-G or G-U base pair.
    Output:
        A dot-bracket string denoting the secondary structure.
    """
    n = len(x)
    x += "#"
    V = np.zeros((n+1, n+1))
    for d in range(1, n):
        for i in range(1, n-d+1):
            j = i + d
            if x[i] + x[j] in ["AU", "UA"]:
                V[i, j] = V[i + 1, j - 1] + p[0, 0]
            elif x[i] + x[j] in ["CG", "GC"]:
                V[i, j] = V[i + 1, j - 1] + p[1, 0]
            elif x[i] + x[j] in ["GU", "UG"]:
                V[i, j] = V[i + 1, j - 1] + p[2, 0]
            for k in range(i, j):
                V[i, j] = max(V[i, j], V[i, k] + V[k+1, j])
    print(V)

    # Trace back to determine signature directly
    sig = np.zeros((3, 1))
    __nussinov_traceback(V, p, x, sig, 1, n)
    return sig


def __nussinov_traceback(V: np.ndarray, p: np.ndarray, x: str, sig: np.ndarray, i: int, j: int):
    if j <= i:
        return
    elif V[i, j] == V[i, j-1]:
        __nussinov_traceback(V, p, x, sig, i, j-1)
        return
    else:
        found = False
        for k in range(i, j):
            if x[k] + x[j] in ["AU", "UA"] and V[i, j] == V[i, k - 1] + V[k + 1, j - 1] + p[0, 0]:
                sig[0, 0] += 1
                found = True
            elif x[k] + x[j] in ["CG", "GC"] and V[i, j] == V[i, k - 1] + V[k + 1, j - 1] + p[1, 0]:
                sig[1, 0] += 1
                found = True
            elif x[k] + x[j] in ["GU", "UG"] and V[i, j] == V[i, k - 1] + V[k + 1, j - 1] + p[2, 0]:
                sig[2, 0] += 1
                found = True
            if found:
                __nussinov_traceback(V, p, x, sig, i, k-1)
                __nussinov_traceback(V, p, x, sig, k+1, j-1)
                return


def nussinov_signature(x: str, y: str):
    """
    Return the signature corresponding to Nussinov-Jacobson model
    for a given RNA sequence and a compatible secondary structure.

    Input:
        x : a string, denoting the RNA nucleotide sequence.
        y : a dot-bracket string, denoting the RNA secondary structure.
    Output:
        A ndarray of dimension 3, where the three entries correspond to
        - the number of A-U base pairs,
        - the number of C-G base pairs, and
        - the number of U-G base pairs.
    """

    # Sanity check, to make sure y might be a possible structure of x.
    assert len(x) == len(y)
    n = len(x)
    stack = []
    res = np.zeros((3, 1))
    for j in range(n):
        if y[j] == '(':
            stack.append(j)
        elif y[j] == ')':
            i = stack.pop()

            if x[i] + x[j] in ["AU", "UA"]:
                res[0, 0] += 1
            elif x[i] + x[j] in ["CG", "GC"]:
                res[1, 0] += 1
            elif x[i] + x[j] in ["UG", "GU"]:
                res[2, 0] += 1
    return res


#seed = np.random.randint(10000)
# print(seed)
#n = 10
#rng = np.random.default_rng(seed)
#rna = ''.join(rng.choice(['A', 'G', 'C', 'U']) for _ in range(n))
x = "UAUUCUGAUG"
print(x)
#y = "...(())"
#x = "GGGAGGUCGUUACAUCUGGGUAACACCGGUACUGAUCCGGUGACCUCCC"
#y = "((((((((.((((......))))..((((.......)))).))))))))"
p = np.ones((3, 1))
#print(nussinov_signature(x, y))
#print(nussinov(x, p))

cnt = 0

set = set()


def backtrack(q, queue, s, original):
    global cnt
    if not q:
        if not queue:
            cnt += 1
            print(cnt, s, nussinov_signature(original, s))
            set.add(tuple(map(tuple, nussinov_signature(original, s).T)))
        return
    backtrack(q[1:], queue, s + ".", original)
    backtrack(q[1:], queue + [q[0]], s + "(", original)
    if queue:
        d = q[0] + queue[0]
        if d in ["AU", "UA", "GC", "CG", "GU", "UG"]:
            backtrack(q[1:], queue[1:], s + ")", original)


backtrack(x, [], "", x)
print(set)
print(len(set))
