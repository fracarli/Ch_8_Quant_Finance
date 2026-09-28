import numpy as np
from scipy.stats import norm

# S = spot price
# E = exercise price
# r = risk free rate
# D = dividend yield
# T = maturity date
# t = current date
# sigma = volatility

# ---------------------------------------------

def d1_d2(S, E, r, D, T, t, sigma):
    d1 = (np.log(S / E) + (r - D + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    return d1, d2

# ---------------------------------------------
# VALUE: Option Value
# ---------------------------------------------

def value_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return S * np.exp(-D * (T - t)) * norm.cdf(d1) - E * np.exp(-r * (T - t)) * norm.cdf(d2)
    

def value_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return E * np.exp(-r * (T - t)) * norm.cdf(-d2) - S * np.exp(-D * (T - t)) * norm.cdf(-d1)


def value_binary_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return  np.exp(-r * (T - t)) * norm.cdf(d2)

def value_binary_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return  np.exp(-r * (T - t)) * norm.cdf(-d2)


# ---------------------------------------------
# DELTA: Rate of Change of Option Value with Respect to Underlying Price
# ---------------------------------------------
"""
The delta measures the sensitivity of the option's price to a small change
in the underlying asset's price. It means it represents the approximate hedge ratio
needed to neutralize directional market risk.
"""

def delta_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return np.exp(-D * (T - t)) * norm.cdf(d1)

def delta_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return -np.exp(-D * (T - t)) * norm.cdf(-d1)

def delta_binary_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return np.exp(-r * (T - t)) * norm.pdf(d2) / (S * sigma * np.sqrt(T - t))

def delta_binary_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return -np.exp(-r * (T - t)) * norm.pdf(-d2) / (S * sigma * np.sqrt(T - t))

# ---------------------------------------------
# GAMMA: Rate of Change of Delta
# ---------------------------------------------
"""
The gamma measures the rate of change of delta with respect to the
underlying asset's price. It means it shows how stable the hedge is and tells
you how much the delta will adjust as the stock price moves.
"""

def gamma_call(S, E, r, D, T, t, sigma):
  d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
  return (np.exp(-D * (T - t)) * norm.pdf(d1) / (S * sigma * np.sqrt(T - t)))


def gamma_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return np.exp(-D * (T - t)) * norm.pdf(-d1) / (S * sigma * np.sqrt(T - t))


def gamma_binary_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return np.exp(-r * (T - t)) * d1* norm.pdf(d2) / (S ** 2 * sigma ** 2 * np.sqrt(T - t))

def gamma_binary_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return np.exp(-r * (T - t)) * norm.pdf(-d2) / (S ** 2 * sigma ** 2 * np.sqrt(T - t))


# ---------------------------------------------
# THETA: Time Decay
# ---------------------------------------------
"""
The theta measures the sensitivity of the option's price to the passage
of time. It means it reflects the daily time decay, showing how much value the
option loses as it gets closer to its expiration date.
"""

def theta_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return (-S * sigma * np.exp(-D * (T - t)) * norm.pdf(d1) / (2 * np.sqrt(T - t))
            + D * S * np.exp(-D * (T - t)) * norm.cdf(d1)
            - r * E * np.exp(-r * (T - t)) * norm.cdf(d2))

def theta_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return (-S * sigma * np.exp(-D * (T - t)) * norm.pdf(d1) / (2 * np.sqrt(T - t))
            - D * S * np.exp(-D * (T - t)) * norm.cdf(-d1)
            + r * E * np.exp(-r * (T - t)) * norm.cdf(-d2))

def theta_binary_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return (-np.exp(-r * (T - t)) * norm.pdf(d2) * d1 / (2 * np.sqrt(T - t))
            + r * np.exp(-r * (T - t)) * norm.cdf(d2))

def theta_binary_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return (-np.exp(-r * (T - t)) * norm.pdf(-d2) * d1 / (2 * np.sqrt(T - t))
            + r * np.exp(-r * (T - t)) * norm.cdf(-d2))


# ---------------------------------------------
# SPEED: Rate of Change of Gamma
# ---------------------------------------------
"""
The speed measures the rate of change of gamma with respect to the
underlying asset's price. It means it quantifies the third-order sensitivity,
showing how rapidly your gamma risks shift as the underlying asset fluctuates.
"""

def speed_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return -np.exp(-D * (T - t)) * norm.pdf(d1) / (sigma ** 2 * S ** 2 * (T - t)) * (d1 + sigma * np.sqrt(T - t))

    
def speed_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return -np.exp(-D * (T - t)) * norm.pdf(-d1) / (sigma ** 2 * S ** 2 * (T - t)) * (-d1 + sigma * np.sqrt(T - t))

def speed_binary_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return np.exp(-r * (T - t)) * norm.pdf(d2) / (sigma ** 2 * S ** 2 * (T - t)) * (d1 + sigma * np.sqrt(T - t))

def speed_binary_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return np.exp(-r * (T - t)) * norm.pdf(-d2) / (sigma ** 2 * S ** 2 * (T - t)) * (-d1 + sigma * np.sqrt(T - t))

# ---------------------------------------------
# VEGA: Volatility Vega
# ---------------------------------------------
"""
The vega measures the sensitivity of the option's price to changes in the
underlying volatility. It means it indicates how much the option value will
change for a 1% shift in the market's expected volatility.
"""

def vega_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return S * np.sqrt(T - t) * np.exp(-D * (T - t)) * norm.pdf(d1)

def vega_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return S * np.sqrt(T - t) * np.exp(-D * (T - t)) * norm.pdf(-d1)

def vega_binary_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return np.sqrt(T - t) * np.exp(-r * (T - t)) * norm.pdf(d2)

def vega_binary_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return np.sqrt(T - t) * np.exp(-r * (T - t)) * norm.pdf(-d2)

# ---------------------------------------------
# RHO_R: Interest Rate Rho
# ---------------------------------------------
"""
The rho measures the sensitivity of the option's price to changes in the
risk-free interest rate. It means it highlights how shifts in borrowing costs
impact the present value of the option payout.
"""

def rho_r_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return E * (T - t) * np.exp(-r * (T - t)) * norm.cdf(d2)

def rho_r_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return - E * (T - t) * np.exp(-r * (T - t)) * norm.cdf(-d2)

def rho_r_binary_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return -(T - t) * np.exp(-r * (T - t)) * norm.cdf(d2)

def rho_r_binary_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return -(T - t) * np.exp(-r * (T - t)) * norm.cdf(-d2)

# ---------------------------------------------
# RHO_D: Dividend Rho
# ---------------------------------------------
"""
The dividend rho measures the sensitivity of the option's price to changes
in the dividend yield. It means it captures how changes in expected payouts
from the underlying asset affect the option's theoretical value.
"""

def rho_D_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return -(T - t) * S * np.exp(-D * (T - t)) * norm.cdf(d1)

def rho_D_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return -(T - t) * S * np.exp(-D * (T - t)) * norm.cdf(-d1)

def rho_D_binary_call(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return (T - t) * np.exp(-D * (T - t)) * norm.cdf(d1)

def rho_D_binary_put(S, E, r, D, T, t, sigma):
    d1, d2 = d1_d2(S, E, r, D, T, t, sigma)
    return (T - t) * np.exp(-D * (T - t)) * norm.cdf(-d1)