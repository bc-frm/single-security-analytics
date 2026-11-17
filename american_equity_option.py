import numpy as np
from math import log, sqrt, exp, erf

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call(S, K, T, r, q, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

# ----------------------------------------------------------------------
# Corrected binomial tree (high-to-low ordering fixed)
# ----------------------------------------------------------------------
def american_call_binomial(S, K, T, r, q, sigma, N=400):
    if T <= 0:
        return max(S - K, 0.0)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Terminal stock prices (high to low)
    stock = S * u**np.arange(N, -1, -1) * d**np.arange(0, N + 1)
    values = np.maximum(stock - K, 0.0)

    for i in range(N - 1, -1, -1):
        # Continuation (note the order: higher node is values[:-1])
        values = discount * (p * values[:-1] + (1 - p) * values[1:])
        # Rebuild stock prices for this layer
        stock = S * u**np.arange(i, -1, -1) * d**np.arange(0, i + 1)
        values = np.maximum(values, stock - K)          # American
    return values[0]


def european_call_binomial(S, K, T, r, q, sigma, N=400):
    if T <= 0:
        return max(S - K, 0.0)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    stock = S * u**np.arange(N, -1, -1) * d**np.arange(0, N + 1)
    values = np.maximum(stock - K, 0.0)

    for i in range(N - 1, -1, -1):
        values = discount * (p * values[:-1] + (1 - p) * values[1:])
    return values[0]

# ----------------------------------------------------------------------
# Thomas algorithm
# ----------------------------------------------------------------------
def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)
    x = np.zeros(n)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n - 1):
        den = b[i] - a[i - 1] * c_[i - 1]
        c_[i] = c[i] / den
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / den
    den = b[n - 1] - a[n - 2] * c_[n - 2]
    d_[n - 1] = (d[n - 1] - a[n - 2] * d_[n - 2]) / den
    x[n - 1] = d_[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x

# ----------------------------------------------------------------------
# Crank-Nicolson (European + American)
# ----------------------------------------------------------------------
def crank_nicolson_call(S0, K, T, r, q, sigma, american=False,
                        M=500, N=500, Smax_factor=4.0):
    if T <= 0:
        return max(S0 - K, 0.0)

    Smax = Smax_factor * max(S0, K)
    dS = Smax / M
    dt = T / N
    S = np.linspace(0.0, Smax, M + 1)
    V = np.maximum(S - K, 0.0)
    payoff = V.copy()

    j = np.arange(1, M)
    alpha = 0.25 * dt * (sigma**2 * j**2 - (r - q) * j)
    beta  = 0.5  * dt * (sigma**2 * j**2 + r)
    gamma = 0.25 * dt * (sigma**2 * j**2 + (r - q) * j)

    a_left = -alpha[1:]
    b_left = 1.0 + beta
    c_left = -gamma[:-1]

    for n in range(N - 1, -1, -1):
        d = alpha * V[0:M-1] + (1.0 - beta) * V[1:M] + gamma * V[2:M+1]

        tau = (N - n) * dt
        boundary_0 = 0.0
        boundary_M = Smax * np.exp(-q * tau) - K * np.exp(-r * tau)

        d[0]  += alpha[0]  * boundary_0
        d[-1] += gamma[-1] * boundary_M

        V[1:M] = thomas_algorithm(a_left, b_left, c_left, d)
        V[0] = boundary_0
        V[M] = boundary_M

        if american:
            V = np.maximum(V, payoff)

    # interpolate
    idx = np.searchsorted(S, S0)
    if idx == 0:
        return V[0]
    if idx >= M:
        return V[M]
    w = (S0 - S[idx - 1]) / dS
    return (1.0 - w) * V[idx - 1] + w * V[idx]

# ----------------------------------------------------------------------
# Greeks (bump-and-revalue)
# ----------------------------------------------------------------------
def compute_greeks(pricer, S, K, T, r, q, sigma, **kw):
    V = pricer(S, K, T, r, q, sigma, **kw)
    ds, dvol, dr, dt = S * 0.001, 0.001, 0.0001, 1.0 / 365.0

    V_up   = pricer(S + ds, K, T, r, q, sigma, **kw)
    V_down = pricer(S - ds, K, T, r, q, sigma, **kw)
    delta  = (V_up - V_down) / (2 * ds)
    gamma  = (V_up - 2 * V + V_down) / ds**2

    V_vup = pricer(S, K, T, r, q, sigma + dvol, **kw)
    V_vdn = pricer(S, K, T, r, q, sigma - dvol, **kw)
    vega  = (V_vup - V_vdn) / (2 * dvol) / 100.0

    V_rup = pricer(S, K, T, r + dr, q, sigma, **kw)
    V_rdn = pricer(S, K, T, r - dr, q, sigma, **kw)
    rho   = (V_rup - V_rdn) / (2 * dr) / 100.0

    V_lat = pricer(S, K, max(T - dt, 0.0), r, q, sigma, **kw)
    theta = (V_lat - V) / dt / 365.0

    return dict(price=V, delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta)

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    S0, K, T = 7417.0, 7400.0, 0.5
    r, q, sigma = 0.0515, 0.0107, 0.16

    print("Analytical European (BS):", round(bs_call(S0, K, T, r, q, sigma), 4))

    print("\n=== Binomial tree ===")
    g_bin = compute_greeks(american_call_binomial, S0, K, T, r, q, sigma, N=400)
    eu_bin = european_call_binomial(S0, K, T, r, q, sigma, N=400)
    print(f"American: {g_bin['price']:.4f}")
    print(f"European: {eu_bin:.4f}")
    print(f"Premium:  {g_bin['price'] - eu_bin:.4f}")
    print(f"Delta:    {g_bin['delta']:.6f}")
    print(f"Gamma:    {g_bin['gamma']:.8f}")
    print(f"Vega:     {g_bin['vega']:.4f}")
    print(f"Rho:      {g_bin['rho']:.4f}")
    print(f"Theta:    {g_bin['theta']:.4f}")

    print("\n=== Crank-Nicolson ===")
    g_cn = compute_greeks(lambda *a, **k: crank_nicolson_call(*a, american=True, **k),
                          S0, K, T, r, q, sigma, M=500, N=500)
    eu_cn = crank_nicolson_call(S0, K, T, r, q, sigma, american=False, M=500, N=500)
    print(f"American: {g_cn['price']:.4f}")
    print(f"European: {eu_cn:.4f}")
    print(f"Premium:  {g_cn['price'] - eu_cn:.4f}")
    print(f"Delta:    {g_cn['delta']:.6f}")
    print(f"Gamma:    {g_cn['gamma']:.8f}")
    print(f"Vega:     {g_cn['vega']:.4f}")
    print(f"Rho:      {g_cn['rho']:.4f}")
    print(f"Theta:    {g_cn['theta']:.4f}")