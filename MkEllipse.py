#! /usr/bin/env python3
#
# Calculate a water parcel's position at each sampling period
# for a given time in a wave train.
#
# Inputs are:
#  depth -- water depth in meters
#  zbar  -- depth in meters to calculate ellipse for, 
#           zbar=0 is at the surface, zbar>0 below the surface
#  df -- Output of mkWave
#  t  -- times to sample at
#
# The output is:
#  a["t"] time of observation starting at zero in seconds
#          The following are in the wave's coordinate frame
#  a["wx"] parcel's displacement in meters along the direction of travel
#  a["wz"] parcel's vertical displacement in meters
#  a["wvx"] parcel's velocity in meters/sec along the direction of travel
#  a["wvz"] parcel's velocity in meters/sec in the vertical direction
#  a["wax"] parcel's acceleration in meters/sec/sec along the direction of travel
#  a["waz"] parcel's acceleration in meters/sec/sec in the vertical direction
#  a["wr"] parcel's distance in meters from (0,0)
#  a["wvPerp"]  parcel's velocity in meters/sec perpendicular to wr
#  a["wOmega"] parcel's angular velocity in radians/sec
#  a["wAlpha"] parcel's angular acceleration in the wy direction in radians/sec/sec
#
#           The following are in earth coordinates with y in the true north direction
#  a["x"]  is the parcel's position in the eastward direction in meters
#  a["y"]  is the parcel's position in the northward direction in meters
#  a["z"]  is the parcel's position in the vertical direction in meters
#  a["vx"]  is the parcel's velocity in the eastward direction in meters/sec
#  a["vy"]  is the parcel's velocity in the northward direction in meters/sec
#  a["vz"]  is the parcel's velocity in the vertical direction in meters/sec
#  a["ax"]  is the parcel's acceleration in the eastward direction in meters/sec/sec
#  a["ay"]  is the parcel's acceleration in the northward direction in meters/sec/sec
#  a["az"]  is the parcel's acceleration in the vertical direction in meters/sec/sec
#  a["omegax"]  is the parcel's angular velocity in the eastward direction in radians/sec
#  a["omegay"]  is the parcel's angular velocity in the northward direction in radians/sec
#  a["omegaz"]  is the parcel's angular velocity in the vertical direction in radians/sec
#  a["alphax"]  is the parcel's angular acceleration in the eastward direction in radians/sec/sec
#  a["alphay"]  is the parcel's angular acceleration in the northward direction in radians/sec/sec
#  a["alphaz"]  is the parcel's angular acceleration in the vertical direction in radians/sec/sec
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os.path
import sys

def myInterp(x, y, t) -> np.array:
    a = interp1d(x, y, kind="quadratic", copy=False, fill_value="extrapolate", assume_sorted=True)
    return a(t)

def mkEllipse(depth:float,
        zbar:float,
        df:pd.DataFrame, 
        t:np.array) -> pd.DataFrame:
    # See http://web.mit.edu/13.021/demos/lectures/lecture19.pdf
    # Particle Orbits at the surface
    # N.B. x in the MIT lecture notes is along the wave's travel direction
    #      y is the vertical direction (I will translate y to z here)

    # A random phase for this wave
    phase = np.random.uniform(0, 2*np.pi) # phase offset for this wave, [0,2pi)

    df["t"] = df.period.cumsum() - (df.period[0]/2) # Source time stamps

    # Interpolate smoothly over t
    period = myInterp(df.t, df.period, t)
    amp = myInterp(df.t, df.amp, t)
    hdg = myInterp(df.t, df.hdg, t)
    waveLen = myInterp(df.t, df["lambda"], t) # Lambda is a reserved keyword
    spd = myInterp(df.t, df.spd, t)
    a = myInterp(df.t, df.a, t)
    b = myInterp(df.t, df.b, t)

    k = 2 * np.pi / waveLen # Wave Number
    omega = 2 * np.pi / period # Angular frequency
    denom = np.sinh(k * depth)

    x0 = -amp * np.cosh(k * (depth + zbar)) / denom
    z0 =  amp * np.sinh(k * (depth + zbar)) / denom

    a = pd.DataFrame({"t": t})

    # The w prefix indicates in wave coordinates, i.e. wx -> Along wave, wz -> vertical
    # sin/cos(-w(t+dt))
    w    = -omega # Negative angular frequency at each time t
    term = w * t + phase # sine/cosine argument
    sterm = np.sin(term)
    cterm = np.cos(term)

    # The position at time t
    a["wx"] = x0 * sterm # Horizontal position along the wave direction
    a["wz"] = z0 * cterm # Vertical position

    # the velocity at time t, i.e. first derivative wrt t
    a["wvx"] =  x0 * w * cterm # d(wx)/dt
    a["wvz"] = -z0 * w * sterm # d(wz)/dt

    # the acceleration at time t, the second derivative wrt t
    a["wax"] = -x0 * w * w * sterm # d^2(wx)/d^2t
    a["waz"] = -z0 * w * w * cterm # d^2(wz)/d^2t

    # the angular velocity
    # omega = (r cross v) / |r|^2
    #       = perpendicular velocity over |r|
    # r = (wx,  0, wz)
    # v = (wvx, 0, wvz)
    # |r| = sqrt(wx^2 + wz^2)
    # vPerp = |r cross v| / |r|
    r = np.sqrt(np.square(a["wx"]) + np.square(a["wz"])) # radial distance
    a["wr"] = r # radial distance
    a["wvPerp"] = (a["wz"] * a["wvx"] - a["wx"] * a["wvz"]) / r # (r cross v) / r
    a["wOmega"] = a["wvPerp"] / r # Angular velocity in wave space in wy direction

    # the angular accceleration
    # a = (wax, 0, waz)
    # alpha = d((r cross v) / |r|^2) / dt
    #       = (r cross a)/|r|^2 - 2/|r| omega d|r|/dt
    # d|r|/dt = d((wx^2 + wz^2)^1/2)/dt
    #         = 1/2 (wx^2 + wz^2)^(-1/2) d(wx^2 + wz^2)/dt
    #         = 1/(2r) (2 wx d(wx)/dt + 2 wz d(wz)/dt)
    #         = (wx wvx + wz wvz) / r
    # since 2D I only need to calculate y component
    alpha0 = (a["wz"] * a["wax"] - a["wx"] * a["waz"]) / np.square(r) # (r crossa) / r^2
    drdt = (a["wx"] * a["wvx"] + a["wz"] * a["wvz"]) / r # dr/dt
    alpha1 = -2/r * a["wOmega"] * drdt # 2/|r| omega d|r|/dt
    a["wAlpha"] = alpha0 + alpha1 # angular acceleration in wy direction

    a["hdg"] = hdg # wave direction in degrees at time t
    return a

def saveCSV(fn:str, name:str, df:pd.DataFrame) -> None:
    (prefix, suffix) = os.path.splitext(fn)
    ofn = "{}.{}.ellipse.csv".format(prefix, name)
    df.to_csv(ofn, index=False)

def plotit(ax, df, x, y):
    ax.plot(df[x], df[y])
    ax.grid()
    ax.set_xlabel(x)
    ax.set_ylabel(y)

if __name__ == "__main__":
    import argparse
    import time
    import MyYAML
    import MyLogger
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", metavar="fn.yml", help="YAML file(s) to load")
    parser.add_argument("wave", metavar="fn.csv", help="Output of mkWave to load")
    parser.add_argument("--seed", type=int, default=int(time.time()),
            help="Random number generator seed, 32 bit int")
    parser.add_argument("--save", action="store_true", help="Should CSV output files be generated?")
    parser.add_argument("--plot", action="store_true", help="Diagnostic plots")
    MyLogger.addArgs(parser)
    args = parser.parse_args()

    logger = MyLogger.mkLogger(args, __name__, "%(asctime)s: %(levelname)s - %(message)s")

    data = MyYAML.load(args.yaml, logger)
    info = pd.read_csv(args.wave)

    # Get the wave name from the wave csv filename, *.name.info.csv
    parts = args.wave.split(".")
    name = parts[-3] # Wave name

    logger.info("Seed=%s", args.seed)
    np.random.seed(seed=args.seed)
    dt = 1 / data["samplingRate"] # Time between observations in seconds
    t = np.arange(0, data["duration"] + dt/2, dt) # Observation times [0, duration]
    df = mkEllipse(data["depth"], data["gliderDepth"], info, t)
    logger.info("Wave %s\n%s", name, df)
    if args.save:
        saveCSV(args.yaml, name, df)

    if args.plot:
        (figs, ax) = plt.subplots(3, 2, figsize=(10,10))
        plotit(ax[0,0], df, "wx", "wz")
        plotit(ax[1,0], df, "wvx", "wvz")
        plotit(ax[2,0], df, "wax", "waz")
        plotit(ax[0,1], df, "t", "wr")
        plotit(ax[1,1], df, "t", "wvPerp")
        plotit(ax[2,1], df, "t", "wOmega")
        plt.show()
