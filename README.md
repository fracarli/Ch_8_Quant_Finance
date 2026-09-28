# Black-Scholes Derivatives & Greeks Engine

A robust, professional-grade Python library for financial derivatives pricing, risk sensitivity analysis (Greeks), and implied volatility calculations. This project is heavily inspired by the mathematical frameworks and notation found in Paul Wilmott's work on quantitative finance (specifically Chapter 8).

---

## Project Features

- **Pricing Engine**: Supports European Plain Vanilla (Call/Put) and Binary options under the Black-Scholes-Merton model with continuous dividend yields.
- **Comprehensive Greeks**: Analytical formulas for Delta, Gamma, Theta, Vega, Speed, and Rho (with respect to both the risk-free rate $r$ and dividend yield $D$).
- **Implied Volatility**: Newton-Raphson root-finding algorithm leveraging option Vega to accurately back out market-implied volatility.
- **Validation & Visualization**: Automated Put-Call Parity verification and a clean `matplotlib` visualization suite for risk profiles.

---

## Project Structure

```text
financial-derivatives/
│
├── src/
│   ├── greeks.py                    # Core pricing formulas and Greeks definitions
│   └── volatility_newton_raphson.py # Implied volatility root-finding solver
│
├── main.py                          # Execution script for validation and plotting
└── README.md                        # Project documentation# Ch_8_Quant_Finance
