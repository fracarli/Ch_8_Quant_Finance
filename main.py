# =====================================================================
# BLACK-SCHOLES & GREEKS EVALUATION SCRIPT
# Author: Francesco Carli
# Date: 28/09/2026
# =====================================================================

import matplotlib.pyplot as plt
import numpy as np
from src.greeks import (
    delta_call,
    gamma_call,
    theta_call,
    value_call,
    value_put,
    vega_call,
)
from src.volatility_NewthonRaphson import implied_volatility_call

if __name__ == "__main__":
  # --- 1. CONFIGURATION & TEST PARAMETERS ---
  S_val = 100.0  # Spot Price
  E_val = 100.0  # Strike Price
  r_val = 0.0  # Risk-Free Rate
  D_val = 0.0  # Dividend Yield
  T_val = 1.0  # Maturity (Years)
  t_val = 0.0  # Current Time
  sigma_val = 0.20  # Volatility (20%)

  print("=" * 50)
  print("          BLACK-SCHOLES MODEL CONFIGURATION          ")
  print("=" * 50)
  print(f"  Spot Price (S)          : {S_val:.2f}")
  print(f"  Strike Price (E)        : {E_val:.2f}")
  print(f"  Risk-Free Rate (r)      : {r_val:.2f}")
  print(f"  Dividend Yield (D)      : {D_val:.2f}")
  print(f"  Maturity (T)            : {T_val:.2f}")
  print(f"  Current Time (t)        : {t_val:.2f}")
  print(f"  Base Volatility (sigma) : {sigma_val:.2f}")
  print("-" * 50)

  # --- 2. THEORETICAL PRICING & PUT-CALL PARITY ---
  c_price = value_call(S_val, E_val, r_val, D_val, T_val, t_val, sigma_val)
  p_price = value_put(S_val, E_val, r_val, D_val, T_val, t_val, sigma_val)

  lhs = c_price - p_price
  rhs = S_val * np.exp(-D_val * T_val) - E_val * np.exp(-r_val * T_val)
  parity_diff = abs(lhs - rhs)

  print("         BLACK-SCHOLES PRICING & PARITY CHECK        ")
  print("=" * 50)
  print(f"  Call Price              : {c_price:.4f}")
  print(f"  Put Price               : {p_price:.4f}")
  print(f"  Put-Call Parity Diff    : {parity_diff:.4e}")
  print("-" * 50)

  # --- 3. IMPLIED VOLATILITY (NEWTON-RAPHSON) ---
  market_target = 10.45
  implied_vol = implied_volatility_call(
      market_target, S_val, E_val, r_val, D_val, T_val, t_val, sigma_init=0.3
  )
  recalculated_price = value_call(
      S_val, E_val, r_val, D_val, T_val, t_val, implied_vol
  )

  print("         IMPLIED VOLATILITY (NEWTON-RAPHSON)         ")
  print("=" * 50)
  print(f"  Target Market Price     : {market_target:.4f}")
  print(f"  Calculated Implied Vol  : {implied_vol:.4f}")
  print(f"  Price via Implied Vol   : {recalculated_price:.4f}")
  print("=" * 50)

  # --- 4. GREEKS SENSITIVITY COMPUTATION ACROSS SPOT RANGE ---
  spot_range = np.linspace(60, 140, 200)

  deltas = [
      delta_call(s, E_val, r_val, D_val, T_val, t_val, sigma_val)
      for s in spot_range
  ]
  gammas = [
      gamma_call(s, E_val, r_val, D_val, T_val, t_val, sigma_val)
      for s in spot_range
  ]
  vegas = [
      vega_call(s, E_val, r_val, D_val, T_val, t_val, sigma_val)
      for s in spot_range
  ]
  thetas = [
      theta_call(s, E_val, r_val, D_val, T_val, t_val, sigma_val)
      for s in spot_range
  ]

  # --- 5. PLOTTING VISUALIZATION ---
  fig, axs = plt.subplots(2, 2, figsize=(12, 10))
  fig.suptitle(
      "Black-Scholes Greeks Analysis (Call Option)",
      fontsize=14,
      fontweight="bold",
  )

  # Delta Plot
  axs[0, 0].plot(spot_range, deltas, color="royalblue", lw=2)
  axs[0, 0].axvline(
      E_val, color="crimson", linestyle="--", label=f"Strike ({E_val})"
  )
  axs[0, 0].set_title("Call Delta")
  axs[0, 0].set_xlabel("Spot Price (S)")
  axs[0, 0].set_ylabel("Delta")
  axs[0, 0].grid(True, alpha=0.3)
  axs[0, 0].legend()

  # Gamma Plot
  axs[0, 1].plot(spot_range, gammas, color="forestgreen", lw=2)
  axs[0, 1].axvline(
      E_val, color="crimson", linestyle="--", label=f"Strike ({E_val})"
  )
  axs[0, 1].set_title("Call Gamma")
  axs[0, 1].set_xlabel("Spot Price (S)")
  axs[0, 1].set_ylabel("Gamma")
  axs[0, 1].grid(True, alpha=0.3)
  axs[0, 1].legend()

  # Vega Plot
  axs[1, 0].plot(spot_range, vegas, color="purple", lw=2)
  axs[1, 0].axvline(
      E_val, color="crimson", linestyle="--", label=f"Strike ({E_val})"
  )
  axs[1, 0].set_title("Call Vega")
  axs[1, 0].set_xlabel("Spot Price (S)")
  axs[1, 0].set_ylabel("Vega")
  axs[1, 0].grid(True, alpha=0.3)
  axs[1, 0].legend()

  # Theta Plot
  axs[1, 1].plot(spot_range, thetas, color="darkorange", lw=2)
  axs[1, 1].axvline(
      E_val, color="crimson", linestyle="--", label=f"Strike ({E_val})"
  )
  axs[1, 1].set_title("Call Theta")
  axs[1, 1].set_xlabel("Spot Price (S)")
  axs[1, 1].set_ylabel("Theta")
  axs[1, 1].grid(True, alpha=0.3)
  axs[1, 1].legend()

  plt.tight_layout()
  plt.show()

# =====================================================================
# APPENDIX: INTUITIONS ON BLACK-SCHOLES PARAMETER SENSITIVITIES
# =====================================================================
"""
1. Spot Price (S) and Strike Price (E):
   - Higher S: Increases Call value (Delta approaches 1) and decreases Put value.
   - Higher E: Decreases Call value (option becomes out-of-the-money) and increases Put value.

2. Volatility (sigma) and Time to Maturity (T):
   - Higher sigma: Increases the value of both Call and Put options (higher uncertainty raises option premium and Vega).
   - Higher T: Increases option value by giving the underlying more time to move favorably; shorter T accelerates time decay (Theta).

3. Risk-Free Rate (r) and Dividend Yield (D):
   - Higher r: Increases Call value (reduces the present value of strike E) and decreases Put value.
   - Higher D: Decreases Call value (underlying expected payout drops) and increases Put value.
"""