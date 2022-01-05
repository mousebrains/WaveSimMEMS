#! /usr/bin/env python3
#
# Analyize MEMs sensor observations to generate Wave Observables such as:
#  Significant Wave Height
#  Dominant period
#  non-directional period
#  directional period

import sys
import os
import logging
import math
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def analyze(fn:str, data:dict, pos:pd.DataFrame, depth:float,
        logger:logging.Logger, nPerSegment:int=None, window:str="boxcar",
        nWide:int=None) -> pd.DataFrame:
    info = calcFFTs(pos=pos, nPerSeg=nPerSegment, logger=logger, window=window, nWide=nWide)
    info = calcCoefs(info)
    summary = calcNonDirectional(info)
    logger.info("Summary Analysis\n:%s", summary)
    return (info, summary)

def calcFFTs(pos:pd.DataFrame, logger:logging.Logger,
        nPerSeg:int=None, window:str="boxcar", scaling:str="spectrum",
        nWide:int=None) -> pd.DataFrame:
    # See paper by Gorman for details on what is done here
    # Retaining just the n<3 terms we have
    # z(t) = a0/2 + a1 cos(2pi f t) + b1 sin(2pi f t) + a2 cos(4pi f t) + b2 sin(4pi f t)
    # For positions we have
    # x(t) = z(t) sin(theta)
    # y(t) = z(t) cos(theta)
    # For velocity we have
    # v_x(t) = v_z(t) cos(theta)
    # v_y(t) = v_z(t) sin(theta)

    dt = np.mean(np.diff(pos.t)) # Sampling period
    fs = 1 / dt # Sampling frequency

    # Use a direct approach to calculating the FFT, then calculate the PSD
    # The notation is from Gorman, Procedia IUTAM 26 (2018) 81-91
    win = signal.get_window(window, pos.x.size) # Signal weighting window
    logging.info("Window %s length %s", window, pos.x.size)

    df = pd.DataFrame() # Return results as columns in df
    df["freq"] = np.fft.rfftfreq(pos.x.size, d=dt) # Single sided bin frequency centers
    # Multipy by dt to get 1/Hz units
    df["X"] = dt * np.fft.rfft(pos.x * win) # FFT of x position, m/Hz
    df["Y"] = dt * np.fft.rfft(pos.y * win) # FFT of y position, m/Hz
    df["Z"] = dt * np.fft.rfft(pos.z * win) # FFT of z position, m/Hz

    df = df[df.freq > 0] # Drop frequency<=0

    if nWide is not None: # Consolidate bins
        newShape = np.array([df.freq.size / nWide, nWide], dtype=int)
        xx = pd.DataFrame()
        xx["freq"] = df.freq.to_numpy().reshape(newShape).mean(axis=1)
        xx["X"] = df.X.to_numpy().reshape(newShape).sum(axis=1)
        xx["Y"] = df.Y.to_numpy().reshape(newShape).sum(axis=1)
        xx["Z"] = df.Z.to_numpy().reshape(newShape).sum(axis=1)
        df = xx

    norm = win.sum() # For boxcar this is L
    logging.info("Window %s length %s norm %s", window, win.size, norm)

    # N.B. Czz = Cxx + Cyy from cos^2 + sin^2 = 1
    df["Cxx"] = np.real(df.X * np.conjugate(df.X)) / (norm * dt) # m^2/Hz
    df["Cyy"] = np.real(df.Y * np.conjugate(df.Y)) / (norm * dt) # m^2/Hz
    df["Czz"] = np.real(df.Z * np.conjugate(df.Z)) / (norm * dt) # m^2/Hz
    df["Qzx"] = np.imag(df.Z * np.conjugate(df.X)) / (norm * dt) # m^2/Hz
    df["Qzy"] = np.imag(df.Z * np.conjugate(df.Y)) / (norm * dt) # m^2/Hz
    df["Cxy"] = np.real(df.X * np.conjugate(df.Y)) / (norm * dt) # m^2/Hz

    return df

def calcCoefs(df:pd.DataFrame) -> pd.DataFrame:
    # From Gorman, only retain n<3 components
    negTerm = 1 # +1 for positions -1 for velocities

    df["a0"] = df.Czz # This is a0 of the Fourier series
    # The following terms will be multiplied by Czz to get the directional
    # spectrum. Since Czz = Czx+Czy, we could just use Czz.
    # But due to noisyness use the following:
    denom = np.sqrt(df.Czz * (df.Cxx + df.Cyy)) # equivalent to Czz in a perfect world
    df["a1"] = negTerm * df.Qzy / denom # Gorman eqn 18
    df["b1"] = negTerm * df.Qzx / denom # Gorman eqn 19
    df["a2"] = (df.Cyy - df.Cxx) / denom     # Gorman eqn 20
    df["b2"] = 2 * df.Cxy / denom # Gorman eqn 21
    return df

def calcNonDirectional(df:pd.DataFrame) -> pd.DataFrame:
    # From the NOAA buoy paper: The notation is:
    # 1 -> Vertical/Z, 2 -> Eastward/X, 3 -> Northward/Y
    # Calculate significant wave heights
    info = pd.Series(dtype=float)
    info["dfreq"] = np.mean(np.diff(df.freq))
    info["m0"] = df.Czz.sum() * info.dfreq
    info["m1"] = (df.Czz * df.freq).sum() * info.dfreq
    info["m2"] = (df.Czz * np.square(df.freq)).sum() * info.dfreq
    info["Hm0"] = 4 * np.sqrt(info.m0)
    info["Tav"] = info.m0 / info.m1
    info["Tzero"] = np.sqrt(info.m0 / info.m2)
    info["Tp"] = 1/df.freq[np.argmax(df.Czz)]
    
    return info

def saveCSV(fn:str, df:pd.DataFrame, item:str) -> None:
    (prefix, suffix) = os.path.splitext(fn)
    ofn = "{}.{}.csv".format(prefix, item)
    df.to_csv(ofn, index=isinstance(df, pd.Series)) # Keep the indices for series

if __name__ == "__main__": # Read in a csv file and call analyze
    import argparse
    import MyYAML
    import MyLogger

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", metavar="fn.yml", help="YAML file to load")
    parser.add_argument("csv", metavar="fn.csv", help="CSV file to load")
    parser.add_argument("--seed", type=int, help="Random number generator seed, 32 bit int")
    parser.add_argument("--save", action="store_true", help="Should CSV output files be generated?")
    parser.add_argument("--plot", action="store_true", help="Plot z versus t")
    MyLogger.addArgs(parser)
    args = parser.parse_args()

    logger = MyLogger.mkLogger(args, __name__, "%(asctime)s: %(levelname)s - %(message)s")

    data = MyYAML.load(args.yaml, logger)
    pos = pd.read_csv(args.csv)

    (output, summary) = analyze(args.csv, data, pos, data["depth"], logger)

    if args.save:
        saveCSV(args.csv, output, "analysis")
        saveCSV(args.csv, summary, "summary")

    if args.plot:
        import MyPolar
        MyPolar.plotit(output)
