def step_towards_u(y, v, lamb):
    n, m = y.shape
    u = np.zeros((n,1))
    for i in range(n):
        a = 0
        b = 0
        for j in range(m):
            if y[i, j] is not None:
                a += v[j]*y[i, j]
                b += v[j]**2
        u[i] = float(a/(b+lamb))
    return u