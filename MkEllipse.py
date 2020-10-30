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
#  rs -- random number generator
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
import os.path

def mkEllipse(depth:float,
        zbar:float,
        df:pd.DataFrame, 
        t:np.array, 
        rs:np.random.RandomState) -> pd.DataFrame:
    # See http://web.mit.edu/13.021/demos/lectures/lecture19.pdf
    # Particle Orbits at the surface
    # N.B. x in the MIT lecture notes is along the wave's travel direction
    #      y is the vertical direction (I will translate y to z here)

    # Start the wave train with a random time/phase offset
    dt0 = df["period"][0] * rs.uniform(0, 1) # Random time offset, [0,period)

    amp = df["amp"].to_numpy()
    k = 2 * np.pi / df["lambda"].to_numpy() # Wave Number
    omega = 2 * np.pi / df["period"].to_numpy() # Angular frequency
    denom = np.sinh(k * depth)
    x0 = -amp * np.cosh(k * (depth + zbar)) / denom
    z0 =  amp * np.sinh(k * (depth + zbar)) / denom

    # Which wave to use at t with the random starting time offset
    # This will transition between waves at a phase of zero
    indices = np.searchsorted(df["period"].cumsum(), t + dt0)
    indices[indices >= omega.size] = omega.size - 1

    x0 = x0[indices] # One to many mapping wave to time
    z0 = z0[indices]

    a = pd.DataFrame({"t": t})

    # The w prefix indicates in wave coordinates, i.e. wx -> Along wave, wz -> vertical
    # sin/cos(-w(t+dt))
    w    = -omega[indices] # Negative angular frequency at each time t
    term = w * (t + dt0)   # sine/cosine argument
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

    # Now rotate to earth coordinates
    # hdg is the angle from true north eastward
    #  x -> Eastward
    #  y -> Northward
    #  z -> Vertical (This is the same in earth and wave coordinates

    a["hdg"] = df["hdg"].to_numpy()[indices] # wave direction in degrees at time t
    hdg = np.radians(a["hdg"]) # Heading at t in radians
    shdg = np.sin(hdg)
    chdg = np.cos(hdg)

    # Rotate position from wave to earth
    a["x"]      = a["wx"] * shdg # Eastward
    a["y"]      = a["wx"] * chdg # Northward
    a["z"]      = a["wz"]        # Vertical

    # Rotate velocity from wave to earth
    a["vx"]   = a["wvx"] * shdg # Eastward
    a["vy"]   = a["wvx"] * chdg # Northward
    a["vz"]   = a["wvz"] * chdg # Vertical

    # Rotate acceleration from wave to earth
    a["ax"] = a["wax"] * shdg # Eastward
    a["ay"] = a["wax"] * chdg # Northward
    a["az"] = a["waz"]        # Vertical

    # Note, angular velocity and acceleration are in wy direction,
    # Rotate angular velocity from wave to earth
    a["omegax"] =  a["wOmega"] * chdg # Eastward
    a["omegay"] = -a["wOmega"] * shdg # Northward
    a["omegaz"] = np.zeros(a["wOmega"].shape) # Vertical always zero

    # Rotate angular acceleration from wave to earth
    a["alphax"] =  a["wAlpha"] * chdg # Eastward
    a["alphay"] = -a["wAlpha"] * shdg # Northward
    a["alphaz"] = np.zeros(a["wAlpha"].shape) # Vertical always zero

    return a

def saveCSV(fn:str, name:str, info:pd.DataFrame) -> None:
    (prefix, suffix) = os.path.splitext(fn)
    ofn = "{}.{}.ellipse.csv".format(prefix, name)
    info.to_csv(ofn, index=False)

if __name__ == "__main__":
    import argparse
    import MyYAML
    import MyLogger

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", metavar="fn.yml", help="YAML file(s) to load")
    parser.add_argument("wave", metavar="fn.csv", help="Output of mkWave to load")
    parser.add_argument("--seed", type=int, help="Random number generator seed, 32 bit int")
    parser.add_argument("--save", action="store_true", help="Should CSV output files be generated?")
    MyLogger.addArgs(parser)
    args = parser.parse_args()

    logger = MyLogger.mkLogger(args, __name__, "%(asctime)s: %(levelname)s - %(message)s")

    data = MyYAML.load(args.yaml, logger)
    info = pd.read_csv(args.wave)

    # Get the wave name from the wave csv filename, *.name.info.csv
    parts = args.wave.split(".")
    name = parts[-3] # Wave name

    rs = np.random.RandomState(seed=args.seed)
    dt = 1 / data["samplingRate"] # Time between observations in seconds
    t = np.arange(0, data["duration"] + dt, dt) # Observation times [0, duration]
    df = mkEllipse(data["depth"], data["gliderDepth"], info, t, rs)
    logger.info("Wave %s\n%s", name, df)
    if args.save:
        saveCSV(args.yaml, name, df)

