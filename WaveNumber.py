#! /usr/bin/env python3
#
# Given an array of frequencies and water depths, find the wave number
# using the dispersion relationship.
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import numpy as np

def waveNumber(d:float, f:np.array, **kwargs) -> np.array:
    ''' 
    Find k, the wave number, using the dispersion relationship
    (2 pi f)^2 = g k tanh(k d) =>
    4 pi^2 f^2 / g = k tanh(kd)

    using Newton-Raphson we'll use the derivative, d/dk:
    0 = tanh (k d) + k^2 (1 - tanh(kd)^2)

    use deep and shallow water approximations for an initial guess of k
    k ~= (2 pi f)^2 / g # Deep water
    k ~= 2 pi f / sqrt(g d) # Shallow water
    k

    dis the water depth in meters
    f is the wave frequency in Hz
    g is the gravitational acceleration in m/s^2
    maxIter is the maximum number of iterations for convergence
    threshold is the maximum fractional error to have converged
    '''

    items = {"threshold": 1e-9, "maxIter": 10, "g": 9.80665} # Defaults
    items.update(kwargs) # Update defaults

    k = np.square(2 * np.pi * f) / items["g"] # Deep water approximation
    qShallow = np.abs(np.tanh(k * d)) < 0.75 # Use shallow water initial approximation
    if np.any(qShallow):
        k[qShallow] = 2 * np.pi * f[qShallow] / np.sqrt(items["g"] * d)# Shallow water

    lhs = np.square(2 * np.pi * f) / items["g"]

    for cnt in range(items["maxIter"]):
        tanh = np.tanh(k * d)
        rhs = k * tanh
        f = rhs - lhs
        dfdk = tanh + k * d * (1 - tanh*tanh) # Derivative of rhs
        dk = f / dfdk # adjustment to k
        err = np.abs(dk / k).max() # Maximum fractional error
        k -= dk
        if err < items["threshold"]: break

    return k

def waveLength(d:float, periods:np.array, **kwargs) -> np.array:
    return 2 * np.pi / waveNumber(d=d, f=1/periods, **kwargs)

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=float, metavar="meters", default=100,
            help="Water depth to calculate wave number(s) for")
    parser.add_argument("--periodMin", type=float, metavar="seconds", default=2,
            help="Minimum period to calculate wave number(s) for")
    parser.add_argument("--periodMax", type=float, metavar="seconds", default=30,
            help="Maximum period to calculate wave number(s) for")
    parser.add_argument("--nPeriods", type=int, default=10, help="Number of wave length steps")
    args = parser.parse_args()

    periods = np.linspace(args.periodMin, args.periodMax, args.nPeriods)
    L = waveLength(args.depth, periods)

    print("Water depth ->", args.depth, "meters")
    df = pd.DataFrame()
    df["tau"] = periods
    df["f"] = 1/periods
    df["k"] = 2 * np.pi / L
    df["lambda"] = L
    print(df)
