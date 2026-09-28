# ---------------------------------------------
# IMPLIED VOLATILITY: Newton-Raphson Method
# ---------------------------------------------
"""
Normally, we plug volatility into Black-Scholes to find the option's fair price. 
Implied volatility flips this logic completely: by observing the actual market price 
at which an option is trading, we work backward to find out what level of volatility 
the market is implicitly expecting. It represents the consensus market view of future 
price fluctuations over the remaining life of the option.
"""

"""
Since there is no direct algebraic formula to isolate volatility from the Black-Scholes equation, 
we use the Newton-Raphson numerical method to solve it. This algorithm starts with an initial 
guess for volatility and iteratively hones in on the correct value by comparing our theoretical 
pricing error against the sensitivity of the option (its Vega). The refinement follows the formula:
sigma_{n+1} = sigma_n - (Price(sigma_n) - Market_Price) / Vega(sigma_n)
"""

from scipy.stats import norm
import numpy as np
from src.greeks import value_call, vega_call

def implied_volatility_call(market_price, S, E, r, D, T, t, sigma_init=0.2, tol=1e-6, max_iter=100):
    sigma = sigma_init
    for _ in range(max_iter):
        price = value_call(S, E, r, D, T, t, sigma)
        diff = price - market_price
        
        if abs(diff) < tol:
            return sigma
            
        vega = vega_call(S, E, r, D, T, t, sigma)
        
        if vega == 0.0:
            break
            
        sigma -= diff / vega
        
        if sigma <= 0:
            sigma = 1e-4
            
    return sigma